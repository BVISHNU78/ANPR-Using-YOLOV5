import logging
from flask import Flask, Response, jsonify, render_template, redirect, url_for, request, session
import cv2
import torch
import numpy as np
import pymysql
import time
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort  
import warnings
import csv
import os
import ffmpeg
import webbrowser
import sys
import json
from torch.cuda.amp import autocast
import threading
import re
import requests
import easyocr
#from basicsr.archs.rrdbnet_arch import RRDBNet
#from realesrgan import RealESRGANer

warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, filename="anpr.log")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set CUDA environment variable
os.environ["CUDACXX"] = os.getenv("CUDA_PATH", "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8")

# Set device
device = 'cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("Running on CPU")

# Directories and files
SAVE_DIR = "saved_plates"
API_URL_LIVE = os.path.join(SAVE_DIR, "vehicle_status.csv")
API_URL_IMAGE = os.path.join(os.getcwd(), "saved_plates")
os.makedirs(SAVE_DIR, exist_ok=True)

# Plate CSV log
PLATE_CSV = os.path.join(SAVE_DIR, "plates_log.csv")
if not os.path.exists(PLATE_CSV):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(PLATE_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "track_id", "image_path", "plate_text"])

# Vehicle status CSV
if not os.path.exists(API_URL_LIVE):
    with open(API_URL_LIVE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "status", "number", "timestamp"])

# Initialize YOLO models
bike_model = YOLO("yolov5nu.pt").to(device)
plate_model = YOLO("license_plate_detector.pt").to(device)

# Initialize Real-ESRGAN
#try:
 #   esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
  #  upsampler = RealESRGANer(scale=10, model_path=r'D:\coding\number plate detection\RealESRGAN_x2plus.pth', model=esrgan_model, device=device)
   # logger.info("Real-ESRGAN initialized successfully")
#except Exception as e:
 #   logger.error(f"Failed to initialize Real-ESRGAN: {e}")
upsampler = None

# Initialize trackers and OCR
anpr_tracker = DeepSort(max_age=30, embedder_gpu=torch.cuda.is_available())
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

ROI_FILE = "roi_config.json"

# Global variables
plate_numbers = {}
entry_logged = set()
exit_logged = set()
vehicle_states = {}
saved_ids = set()
success_log = {}
saved_scores = {}
csv_lock = threading.Lock()

def save_roi(cam_name, top_left, bottom_right):
    """Save ROI coordinates to roi_config.json."""
    try:
        with open(ROI_FILE, "r") as f:
            roi_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        roi_data = {}

    roi_data[cam_name] = {"top_left": top_left, "bottom_right": bottom_right}
    with open(ROI_FILE, "w") as f:
        json.dump(roi_data, f, indent=2)

def load_roi(cam_name):
    """Load ROI coordinates from roi_config.json."""
    try:
        with open(ROI_FILE, "r") as f:
            data = json.load(f)
            cam_roi = data.get(cam_name)
            if cam_roi:
                return tuple(cam_roi["top_left"]), tuple(cam_roi["bottom_right"])
    except:
        pass
    return None, None

def apply_clahe(image):
    """Apply CLAHE to enhance contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def is_blurry(image, threshold=20):  # Lowered threshold for low-res crops
    """Check if an image is blurry."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def is_large_enough(crop, min_width=150, min_height=150):
    """Check if cropped image meets minimum size requirements."""
    h, w = crop.shape[:2]
    return w >= min_width and h >= min_height

def plate_present(vehicle_crop):
    """Check if a license plate is present in the vehicle crop."""
    results = plate_model(vehicle_crop)[0]
    return len(results.boxes) > 0

def extract_text_easyocr(image):
    """Extract text from an image using EasyOCR."""
    results = reader.readtext(image)
    text = ''.join([res[1] for res in results])
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def validate_plate(text):
    """Validate license plate text format."""
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    match = re.search(r'[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}', cleaned)
    if match:
        return match.group()
    if 6 <= len(cleaned) <= 10:
        return cleaned
    return "INVALID"

def is_good_plate(crop):
    """Check if the plate crop has acceptable dimensions and aspect ratio."""
    h, w = crop.shape[:2]
    if h < 15 or w < 40:
        return False
    ratio = w / h
    return 1.0 <= ratio <= 6.5

def _log_plate_to_csv(track_id, filepath, plate_text):
    """Log plate data to CSV with thread safety."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with csv_lock:
        with open(PLATE_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, track_id, filepath, plate_text])

def enhance_plate(plate):
    """Apply Real-ESRGAN to enhance plate image."""
    if upsampler is None:
        logger.warning("Real-ESRGAN not available, skipping enhancement")
        return plate
    try:
        enhanced, _ = upsampler.enhance(plate, outscale=2)
        logger.info("Plate image enhanced with Real-ESRGAN")
        return enhanced
    except Exception as e:
        logger.error(f"Real-ESRGAN enhancement failed: {e}")
        return plate

def save_resized_plate(bike_crop, track_id, full_frame=None):
    """
    Detect and save resized license plate images for a given track_id.
    Applies CLAHE and Real-ESRGAN for low-resolution crops.
    """
    logger.info(f"Processing crop for track_id {track_id}: {bike_crop.shape}")
    if bike_crop.shape[0] < 312 or bike_crop.shape[1] < 267:
        logger.warning(f"Low-resolution crop detected: {bike_crop.shape}")

    # Apply CLAHE
    bike_crop = apply_clahe(bike_crop)

    results = plate_model(bike_crop, conf=0.4)[0]
    if len(results.boxes) == 0 and full_frame is not None:
        logger.info("No plate in crop, trying full frame")
        full_frame = apply_clahe(full_frame)
        results = plate_model(full_frame, conf=0.3)[0]
    if len(results.boxes) == 0:
        logger.info("[SKIP] No plate detected")
        return None

    plate_folder = os.path.join(SAVE_DIR, str(track_id))
    os.makedirs(plate_folder, exist_ok=True)
    last_fp = None

    for x1, y1, x2, y2, score, cls in results.boxes.data.tolist():
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(bike_crop.shape[1], int(x2)), min(bike_crop.shape[0], int(y2))
        plate = bike_crop[y1:y2, x1:x2]
        if plate.size == 0:
            logger.info("[SKIP] Empty plate crop")
            continue

        if not is_good_plate(plate):
            logger.info("[SKIP] Plate crop not good (size/aspect)")
            continue

        # Enhance low-resolution plates
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        if plate.shape[0] < 50 or plate.shape[1] < 100:
            plate=enhance_plate(plate)
            cv2.imwrite(os.path.join(plate_folder, f"original_{track_id}_{timestamp}.jpg"), plate)
            cv2.imwrite(os.path.join(plate_folder, f"enhanced_{track_id}_{timestamp}.jpg"), plate)

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
        if clarity < 20:
            logger.info("[SKIP] Too blurry")
            continue

        resized = cv2.resize(plate, (500, 200), interpolation=cv2.INTER_CUBIC)
        fname = f"plate_{track_id}_{timestamp}.jpg"
        fp = os.path.join(plate_folder, fname)

        if cv2.imwrite(fp, resized):
            logger.info(f"[SAVED] Plate image: {fp}")
            last_fp = fp
            raw = extract_text_easyocr(resized)
            plate_text = validate_plate(raw)
            logger.info(f"[OCR] Extracted: {plate_text}")
            _log_plate_to_csv(track_id, fp, plate_text)
        else:
            logger.error("[ERROR] Write failed")

    if last_fp is None:
        logger.info("[SKIP] No valid plate after filtering")
    return last_fp

def save_vehicle(vehicle_crop, track_id, status="UNKNOWN"):
    """Save vehicle image into the same folder as plates."""
    from google.cloud import vision
    import pymysql

    folder = os.path.join(SAVE_DIR, str(track_id))
    os.makedirs(folder, exist_ok=True)

    existing_files = [f for f in os.listdir(folder) if f.startswith('vehicle_') and f.endswith('.jpg')]
    if len(existing_files) >= 1:
        logger.info(f"[SKIP] Already 1 vehicle image saved for track_id {track_id}")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"vehicle_{timestamp}.jpg"
    filepath = os.path.join(folder, fname)
    success = cv2.imwrite(filepath, vehicle_crop)

    if not success:
        logger.error(f"[ERROR] Failed to save vehicle image: {filepath}")
        return

    logger.info(f"[SAVED] Vehicle image saved: {filepath}")

    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vk.json'
        client = vision.ImageAnnotatorClient()
        with open(filepath, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)
        texts = response.text_annotations
        ocr_text = [f"\r\n{text.description}" for text in texts]
        text_block = ocr_text[0] if ocr_text else ""
    except Exception as e:
        logger.error(f"[ERROR] Google OCR failed: {e}")
        text_block = ""

    def extract_number_plate(text):
        lines = [line.strip().replace(" ", "").upper() for line in text.strip().splitlines() if line.strip()]
        for i in range(len(lines)):
            line1 = lines[i]
            if re.fullmatch(r'[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}', line1):
                for j in range(1, 6):
                    if i + j < len(lines):
                        line2 = lines[i + j]
                        if re.fullmatch(r'[A-Z0-9]{4,5}', line2):
                            return line1 + line2
        return validate_plate(text)

    plate_number = extract_number_plate(text_block)
    logger.info(f"[OCR] (Vision) Plate Number Detected: {plate_number}")

    _log_plate_to_csv(track_id, filepath, plate_number)

    try:
        db = pymysql.connect(
            host="localhost",
            user=os.getenv("DB_USER", "monsow_vehicle_db"),
            password=os.getenv("DB_PASSWORD", "@ep6I$8PpPg6"),
            database="monsow_vehicle_db"
        )
        cursor = db.cursor()
        sql = """
            INSERT INTO vehicle_logs (image_path, number_plate, status, timestamp)
            VALUES (%s, %s, %s, %s)
        """
        cursor.execute(sql, (filepath, plate_number, status, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        db.commit()
        logger.info(f"[DB] Inserted: {plate_number}, {status}")
        cursor.close()
        db.close()
    except Exception as e:
        logger.error(f"[ERROR] DB insert failed: {e}")

def anpr_stream(cam_name):
    """Stream video with ANPR processing and ROI overlay."""
    cam_data = read_camera_data()
    rtsp_url = cam_data.get(cam_name)
    if not rtsp_url:
        return
    cap = cv2.VideoCapture(rtsp_url, apiPreference=cv2.CAP_FFMPEG)
    retries = 0
    max_retries = 5

    while retries < max_retries:
        ROI_TOP_LEFT, ROI_BOTTOM_RIGHT = load_roi(cam_name)
        if not ROI_TOP_LEFT or not ROI_BOTTOM_RIGHT:
            ROI_TOP_LEFT = (300, 50)
            ROI_BOTTOM_RIGHT = (800, 500)

        ret, frame = cap.read()
        if not ret:
            logger.error(f"[ERROR] Failed to read frame from {cam_name}. Retry {retries+1}/{max_retries}")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, apiPreference=cv2.CAP_FFMPEG)
            time.sleep(2 ** retries)
            retries += 1
            continue
        retries = 0

        height, width = frame.shape[:2]
        results = bike_model(frame)[0]
        detections = []

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = result
            if int(cls) == 3:
                detections.append(([x1, y1, x2 - x1, y2 - y1], score, int(cls)))

        tracks = anpr_tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx, cy = (l + r) // 2, (t + b) // 2
            l, t, r, b = max(0, l), max(0, t), min(width, r), min(height, b)

            if ROI_TOP_LEFT[0] <= cx <= ROI_BOTTOM_RIGHT[0] and ROI_TOP_LEFT[1] <= cy <= ROI_BOTTOM_RIGHT[1]:
                bike_crop = frame[t:b, l:r]
                if bike_crop.shape[0] < 30 or bike_crop.shape[1] < 30:
                    logger.info("[SKIP] Cropped bike image too small to save")
                else:
                    logger.debug("[DEBUG] Saving vehicle and plate image")
                    status = "IN" if "in" in cam_name.lower() else "OUT" if "out" in cam_name.lower() else "UNKNOWN"
                    save_vehicle(bike_crop, track_id, status)
                    save_resized_plate(bike_crop, track_id, full_frame=frame)

            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, b + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if track_id in vehicle_states:
                prev_cy = vehicle_states[track_id]
                if prev_cy < 300 <= cy and track_id not in entry_logged:
                    with csv_lock:
                        with open(API_URL_LIVE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([track_id, "IN", "UNKNOWN", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                    logger.info(f"[LOG] Track {track_id} entered")
                    entry_logged.add(track_id)
                elif prev_cy > 300 >= cy and track_id not in exit_logged:
                    with csv_lock:
                        with open(API_URL_LIVE, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([track_id, "OUT", "UNKNOWN", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                    logger.info(f"[LOG] Track {track_id} exited")
                    exit_logged.add(track_id)

            vehicle_states[track_id] = cy

        cv2.rectangle(frame, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/process_saved_vehicles')
def process_saved_vehicles():
    from google.cloud import vision
    import pymysql

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vk.json'
    client = vision.ImageAnnotatorClient()

    db = pymysql.connect(
        host="localhost",
        user=os.getenv("DB_USER", "monsow_vehicle_db"),
        password=os.getenv("DB_PASSWORD", "@ep6I$8PpPg6"),
        database="monsow_vehicle_db"
    )
    cursor = db.cursor()

    folder_path = SAVE_DIR
    image_extensions = ('.jpg', '.jpeg', '.png')
    count = 0

    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions) and file.startswith("vehicle_"):
                filepath = os.path.join(subdir, file)
                try:
                    with open(filepath, "rb") as image_file:
                        content = image_file.read()
                    image = vision.Image(content=content)
                    response = client.document_text_detection(image=image)
                    texts = response.text_annotations
                    ocr_text = [f"\r\n{text.description}" for text in texts]
                    text_block = ocr_text[0] if ocr_text else ""

                    def extract_number_plate(text):
                        lines = [line.strip().replace(" ", "").upper() for line in text.strip().splitlines() if line.strip()]
                        for i in range(len(lines)):
                            line1 = lines[i]
                            if re.fullmatch(r'[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}', line1):
                                for j in range(1, 6):
                                    if i + j < len(lines):
                                        line2 = lines[i + j]
                                        if re.fullmatch(r'[A-Z0-9]{4,5}', line2):
                                            return line1 + line2
                        return validate_plate(text)

                    plate_number = extract_number_plate(text_block)
                    logger.info(f"[OCR] {file}: {plate_number}")

                    sql = """
                        INSERT INTO vehicle_logs (image_path, number_plate, status, timestamp)
                        VALUES (%s, %s, %s, %s)
                    """
                    values = (filepath, plate_number, "IN", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    cursor.execute(sql, values)
                    db.commit()
                    count += 1

                    _log_plate_to_csv("NA", filepath, plate_number)

                except Exception as e:
                    logger.error(f"[ERROR] Failed to process {file}: {e}")

    cursor.close()
    db.close()
    return f" Processed {count} vehicle image(s) from {SAVE_DIR}."

def resource_path(relative_path):
    """Get absolute path for resources, handling PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

model = YOLO("yolov5n.pt").to(device)
try:
    model.conf = 0.6
    model.iou = 0.5
except Exception:
    pass

valid_classes = ['motorbike', 'motorcycle', 'car', 'truck']

model_path = resource_path("mobilenetv2_bottleneck_wts.pt")
vehicle_tracker = DeepSort(
    embedder="mobilenet",
    embedder_wts=model_path,
    max_iou_distance=0.7,
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_cosine_distance=0.2,
    nn_budget=None,
    gating_only_position=False,
    override_track_class=None,
    half=True,
    bgr=True,
    embedder_gpu=torch.cuda.is_available()
)

line_x = 395
margin = 10
track_positions = {}
track_sides = {}
already_counted = set()
in_count, out_count = 0, 0

def read_rtsp_url():
    """Read RTSP URLs from config file."""
    config_file_path = resource_path('config.txt')
    if not os.path.isfile(config_file_path):
        return []
    with open(config_file_path, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def read_camera_data():
    """Read camera data from cameras.json."""
    try:
        with open('cameras.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def non_max_suppression_fast(boxes, overlapThresh=0.4):
    """Perform non-maximum suppression on bounding boxes."""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        suppress = [last]

        for pos in idxs[:-1]:
            xx1 = max(x1[last], x1[pos])
            yy1 = max(y1[last], y1[pos])
            xx2 = min(x2[last], x2[pos])
            yy2 = min(y2[last], y2[pos])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[pos]

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.setdiff1d(idxs, suppress)

    return boxes[pick].astype("int")

def draw_line(frame, cam_name):
    """Draw a line on the frame based on lines.json."""
    try:
        with open('lines.json', 'r') as f:
            lines = json.load(f)
        line = lines.get(cam_name)
        if line:
            x1, y1 = int(line['x1']), int(line['y1'])
            x2, y2 = int(line['x2']), int(line['y2'])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    except:
        pass
    return frame

def generate_frames(cam_name, rtsp_url):
    """Generate frames for vehicle counting with line crossing."""
    global in_count, out_count, track_positions, track_sides, already_counted
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    csv_file = 'vehicle_log.csv'
    if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['vehicle_id', 'status', 'timestamp'])

    while True:
        success, frame = cap.read()
        if not success:
            logger.error("RTSP stream not available. Attempting to reconnect...")
            cap.release()
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            time.sleep(2)
            continue

        frame = draw_line(frame, cam_name)
        results = model(frame)
        result = results[0]
        detections = result.boxes.data

        dets = []
        raw_boxes = []

        if detections is not None and len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.tolist()
                class_name = model.names[int(cls)]
                if class_name in valid_classes and conf > 0.6:
                    raw_boxes.append([x1, y1, x2, y2])

        if raw_boxes:
            nms_boxes = non_max_suppression_fast(raw_boxes)
            for (x1, y1, x2, y2) in nms_boxes:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                w, h = x2 - x1, y2 - y1
                dets.append(([x1, y1, w, h], 0.8, 'vehicle'))

        tracks = vehicle_tracker.update_tracks(dets, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx = int((l + r) / 2)

            if track_id not in track_positions:
                track_positions[track_id] = []

            track_positions[track_id].append(cx)
            if len(track_positions[track_id]) > 10:
                track_positions[track_id].pop(0)

            prev_positions = track_positions[track_id]
            if len(prev_positions) >= 2:
                movement = prev_positions[-1] - prev_positions[0]
                if abs(movement) > 15:
                    curr_side = "left" if cx < line_x - margin else "right" if cx > line_x + margin else None
                    prev_side = track_sides.get(track_id)

                    if prev_side is None:
                        track_sides[track_id] = curr_side
                    elif curr_side and prev_side != curr_side:
                        if track_id not in already_counted:
                            direction = 'in' if curr_side == 'right' else 'out'
                            if direction == 'in':
                                in_count += 1
                            else:
                                out_count += 1

                            already_counted.add(track_id)
                            track_sides[track_id] = curr_side

                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            vehicle_id = f"ID_{track_id}"

                            with csv_lock:
                                with open(csv_file, 'a', newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerow([vehicle_id, direction, timestamp])

                            logger.info(f"[LOG] {timestamp} - {direction.upper()} - {vehicle_id}")

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f'ID:{track_id}', (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(frame, f'IN: {in_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f'OUT: {out_count}', (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def run_anpr_background():
    """Run ANPR processing in the background for all cameras."""
    cam_data = read_camera_data()

    for cam_name, rtsp_url in cam_data.items():
        cap = cv2.VideoCapture(rtsp_url, apiPreference=cv2.CAP_FFMPEG)
        status_label = "IN" if "in" in cam_name.lower() else "OUT" if "out" in cam_name.lower() else "UNKNOWN"

        while True:
            ROI_TOP_LEFT, ROI_BOTTOM_RIGHT = load_roi(cam_name)
            if not ROI_TOP_LEFT or not ROI_BOTTOM_RIGHT:
                ROI_TOP_LEFT = (400, 100)
                ROI_BOTTOM_RIGHT = (800, 500)

            logger.info(f"[INFO] Background ANPR started for camera: {cam_name}")

            ret, frame = cap.read()
            if not ret:
                logger.error(f"[ERROR] Failed to read frame from {cam_name}. Reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(rtsp_url, apiPreference=cv2.CAP_FFMPEG)
                time.sleep(2)
                continue

            height, width = frame.shape[:2]
            results = bike_model(frame)[0]
            detections = []

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, cls = result
                if int(cls) == 3:
                    detections.append(([x1, y1, x2 - x1, y2 - y1], score, int(cls)))

            tracks = anpr_tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                l, t, r, b = map(int, track.to_ltrb())
                cx, cy = (l + r) // 2, (t + b) // 2
                l, t, r, b = max(0, l), max(0, t), min(width, r), min(height, b)

                if ROI_TOP_LEFT[0] <= cx <= ROI_BOTTOM_RIGHT[0] and ROI_TOP_LEFT[1] <= cy <= ROI_BOTTOM_RIGHT[1]:
                    bike_crop = frame[t:b, l:r]
                    if bike_crop.size > 0:
                        plate_results = plate_model(bike_crop)[0]
                        for plate_box in plate_results.boxes.data.tolist():
                            px1, py1, px2, py2, *_ = map(int, plate_box)
                            px1, py1 = max(0, px1), max(0, py1)
                            px2, py2 = min(bike_crop.shape[1], px2), min(bike_crop.shape[0], py2)
                            plate_crop = bike_crop[py1:py2, px1:px2]

                            if plate_crop.size > 0:
                                clarity_score = cv2.Laplacian(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                                if clarity_score > 20 and 20 < plate_crop.shape[0] < 500 and 50 < plate_crop.shape[1] < 1000:
                                    save_vehicle(bike_crop, track_id, status_label)
                                    save_resized_plate(bike_crop, track_id, full_frame=frame)
                                    break

                if track_id in vehicle_states:
                    prev_cy = vehicle_states[track_id]
                    if prev_cy < 300 <= cy and track_id not in entry_logged:
                        with csv_lock:
                            with open(API_URL_LIVE, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([track_id, "IN", "UNKNOWN", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                        logger.info(f"[LOG] Track {track_id} entered")
                        entry_logged.add(track_id)
                    elif prev_cy > 300 >= cy and track_id not in exit_logged:
                        with csv_lock:
                            with open(API_URL_LIVE, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([track_id, "OUT", "UNKNOWN", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                        logger.info(f"[LOG] Track {track_id} exited")
                        exit_logged.add(track_id)

                vehicle_states[track_id] = cy

            time.sleep(0.05)

@app.route('/', methods=['GET', 'POST'])
def login():
    """Handle login requests."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """Render the dashboard with camera list."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    cam_data = read_camera_data()
    return render_template('dashboard.html', cameras=cam_data)

@app.route('/set_line/<cam_name>')
def set_line(cam_name):
    """Render page to set line for vehicle counting."""
    return render_template('set_line.html', cam_name=cam_name)

@app.route('/save_line/<cam_name>', methods=['POST'])
def save_line(cam_name):
    """Save line coordinates to lines.json."""
    data = request.json
    try:
        with open('lines.json', 'r') as f:
            lines = json.load(f)
    except FileNotFoundError:
        lines = {}

    lines[cam_name] = data
    with open('lines.json', 'w') as f:
        json.dump(lines, f, indent=2)
    return '', 204

@app.route("/set_roi", methods=["POST"])
def set_roi():
    """Save ROI coordinates received from the front-end."""
    data = request.get_json()
    cam = data.get("cam")
    top_left = tuple(data.get("top_left", [0, 0]))
    bottom_right = tuple(data.get("bottom_right", [0, 0]))

    if not cam or not (len(top_left) == 2 and len(bottom_right) == 2):
        return jsonify({"error": "Invalid ROI data"}), 400
    if top_left[0] >= bottom_right[0] or top_left[1] >= bottom_right[1]:
        return jsonify({"error": "Invalid ROI coordinates"}), 400

    save_roi(cam, top_left, bottom_right)
    return jsonify({"message": "ROI saved", "top_left": top_left, "bottom_right": bottom_right})

@app.route("/get_roi/<cam_name>")
def get_roi(cam_name):
    """Fetch existing ROI coordinates for a camera."""
    top_left, bottom_right = load_roi(cam_name)
    if top_left and bottom_right:
        return jsonify({"top_left": top_left, "bottom_right": bottom_right})
    return jsonify({"top_left": [0, 0], "bottom_right": [0, 0]})

@app.route("/draw_roi/<cam_name>")
def set_roi_ui(cam_name):
    """Render page for drawing ROI."""
    return render_template("roi_draw.html", cam_name=cam_name)

@app.route('/anpr/<cam_name>')
def anpr_video(cam_name):
    """Stream ANPR video for a camera."""
    cam_data = read_camera_data()
    rtsp_url = cam_data.get(cam_name)
    if not rtsp_url:
        return f"Camera '{cam_name}' not found in cameras.json", 404
    return Response(anpr_stream(cam_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/anpr_page/<cam_name>')
def anpr_page(cam_name):
    """Render ANPR page for a camera."""
    return render_template('anpr.html', cam_name=cam_name)

@app.route("/video_feed/<cam_name>")
def video_feed(cam_name):
    """Stream video feed for ROI drawing and ANPR."""
    return Response(anpr_stream(cam_name), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/add_camera', methods=['POST'])
def add_camera():
    """Add a new camera to cameras.json."""
    try:
        cam_name = request.form['cam_name'].strip()
        cam_url = request.form['cam_url'].strip()
        if not cam_url.endswith('/video1_stream.m3u8'):
            if not cam_url.endswith('/'):
                cam_url += '/'
            cam_url += 'video1_stream.m3u8'

        cameras = read_camera_data()
        if cam_name in cameras:
            return f"Camera '{cam_name}' already exists.", 400

        cameras[cam_name] = cam_url
        with open('cameras.json', 'w') as f:
            json.dump(cameras, f, indent=4)
        return redirect(url_for('dashboard'))
    except Exception as e:
        logger.error(f"Error in /add_camera: {e}")
        return "Bad Request", 400

@app.route('/camera/<cam_name>')
def camera(cam_name):
    """Render stream page for a camera."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    cam_data = read_camera_data()
    if cam_name not in cam_data:
        return "Camera not found", 404
    return render_template('stream.html', cam_name=cam_name)

@app.route('/delete_camera', methods=['POST'])
def delete_camera():
    """Delete a camera from cameras.json."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    cam_name = request.form.get('cam_name')
    if not cam_name:
        return "Camera name required", 400

    cameras = read_camera_data()
    if cam_name in cameras:
        del cameras[cam_name]
        with open('cameras.json', 'w') as f:
            json.dump(cameras, f, indent=4)
    return redirect(url_for('dashboard'))

@app.route('/edit_camera', methods=['POST'])
def edit_camera():
    """Edit an existing camera in cameras.json."""
    original_name = request.form['original_name']
    new_name = request.form['new_name']
    new_url = request.form['new_url']

    cameras = read_camera_data()
    if original_name != new_name:
        cameras.pop(original_name, None)
    cameras[new_name] = new_url
    with open('cameras.json', 'w') as f:
        json.dump(cameras, f)
    return redirect(url_for('dashboard'))

@app.route('/index')
def index():
    """Redirect to dashboard if logged in."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    """Log out the user."""
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/video/<cam_name>')
def video(cam_name):
    """Stream video for vehicle counting."""
    cam_data = read_camera_data()
    url = cam_data.get(cam_name)
    if not url:
        return "Invalid camera", 404
    return Response(generate_frames(cam_name, url), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/streams')
def streams():
    """Render streams page with RTSP URLs."""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    urls = read_rtsp_url()
    return render_template('streams.html', stream_count=len(urls))

# Start background ANPR processing
threading.Thread(target=run_anpr_background, daemon=True).start()

if __name__ == '__main__':
    webbrowser.open("http://localhost:2086")
    app.run(host="0.0.0.0", port=2086)