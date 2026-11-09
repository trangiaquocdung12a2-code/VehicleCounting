import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# =====================================================
# 1. CẤU HÌNH (CONFIGURATION)
# =====================================================

MODEL_PATH = "yolo11m.pt" 
DAY_VIDEO_PATH = "videos/traffic_day.mp4"
NIGHT_VIDEO_PATH = "videos/traffic_night.mp4"

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# COCO vehicle class IDs
VEHICLE_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

VEHICLE_CLASSES = list(VEHICLE_NAMES.keys())

DAY_CONFIDENCE = 0.25
NIGHT_CONFIDENCE = 0.15 
IMAGE_SIZE = 1088

# Tọa độ vùng đếm (ROI) - Tỷ lệ phần trăm
ROI_POINTS_REL = [
    (0.08, 0.35),
    (0.92, 0.35),
    (0.99, 0.99),
    (0.01, 0.99)
]

LINE_ORIENTATION = "horizontal"
LINE_POSITION_REL = 0.8
MAX_JUMP_REL = 0.20

MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720

# =====================================================
# 2. XỬ LÝ ẢNH BAN ĐÊM (3-STAGE PIPELINE)
# =====================================================

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_night_image(frame):
    frame_gamma = adjust_gamma(frame, gamma=0.7)
    smoothed = cv2.bilateralFilter(frame_gamma, d=5, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    
    enhanced_lab = cv2.merge((lightness, channel_a, channel_b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# =====================================================
# 3. GIAO DIỆN CHỌN NGUỒN VIDEO
# =====================================================

def ask_yes_no(message, default=True):
    answer = input(message).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "c", "co")

def choose_source():
    print("\n======================================")
    print(" YOLO VEHICLE COUNTING")
    print("======================================")
    print("1. Daytime video")
    print("2. Nighttime video")
    print("3. Webcam")
    print("4. Other video")
    print("======================================")

    choice = input("Select option (1-4): ").strip()

    if choice == "1":
        return DAY_VIDEO_PATH, False, "day"
    if choice == "2":
        return NIGHT_VIDEO_PATH, True, "night"
    if choice == "3":
        enhance = ask_yes_no("Enhance low-light image? (y/n, default y): ", True)
        return 0, enhance, "webcam"
    if choice == "4":
        path = input("Enter video path: ").strip().strip('"')
        enhance = ask_yes_no("Enhance low-light image? (y/n, default n): ", False)
        return path, enhance, "custom"

    print("Invalid choice.")
    return None, False, "unknown"

# =====================================================
# 4. LOGIC VÙNG ROI VÀ VẠCH ĐẾM
# =====================================================

def build_roi(width, height):
    points = []
    for relative_x, relative_y in ROI_POINTS_REL:
        x = max(0, min(int(relative_x * width), width - 1))
        y = max(0, min(int(relative_y * height), height - 1))
        points.append([x, y])
    return np.array(points, dtype=np.int32)

def get_line_position(width, height):
    if LINE_ORIENTATION == "horizontal":
        return int(height * LINE_POSITION_REL)
    return int(width * LINE_POSITION_REL)

def is_inside_roi(x, y, roi):
    return cv2.pointPolygonTest(roi, (int(x), int(y)), False) >= 0

def crossed_counting_line(previous_x, previous_y, current_x, current_y, line_position):
    if LINE_ORIENTATION == "horizontal":
        old_position, new_position = previous_y, current_y
    else:
        old_position, new_position = previous_x, current_x

    return (old_position < line_position and new_position >= line_position) or \
           (old_position > line_position and new_position <= line_position)

# =====================================================
# 5. VẼ ĐỒ HỌA (ROI, VẠCH, THỐNG KÊ)
# =====================================================

def draw_road_region(frame, roi, line_position):
    cv2.polylines(frame, [roi], True, (0, 255, 255), 2)
    min_x, max_x = int(np.min(roi[:, 0])), int(np.max(roi[:, 0]))
    min_y, max_y = int(np.min(roi[:, 1])), int(np.max(roi[:, 1]))

    if LINE_ORIENTATION == "horizontal":
        cv2.line(frame, (min_x, line_position), (max_x, line_position), (0, 0, 255), 3)
    else:
        cv2.line(frame, (line_position, min_y), (line_position, max_y), (0, 0, 255), 3)

    cv2.putText(frame, "COUNTING LINE", (min_x + 5, max(25, min_y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def draw_statistics(frame, roi_counts, line_counts, processing_fps, night_mode):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (390, 345), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    lines = [
        ("VEHICLE COUNTING", (0, 255, 255), 0.65, 30),
        ("UNIQUE VEHICLES IN ROI", (0, 255, 0), 0.53, 60),
        (f"Car: {roi_counts['Car']}", (255, 255, 255), 0.48, 87),
        (f"Motorcycle: {roi_counts['Motorcycle']}", (255, 255, 255), 0.48, 112),
        (f"Bus: {roi_counts['Bus']}", (255, 255, 255), 0.48, 137),
        (f"Truck: {roi_counts['Truck']}", (255, 255, 255), 0.48, 162),
        (f"ROI total: {sum(roi_counts.values())}", (0, 255, 0), 0.52, 188),
        ("VEHICLES CROSSING LINE", (0, 165, 255), 0.53, 220),
        (f"Car: {line_counts['Car']}", (255, 255, 255), 0.48, 247),
        (f"Motorcycle: {line_counts['Motorcycle']}", (255, 255, 255), 0.48, 272),
        (f"Bus: {line_counts['Bus']}  Truck: {line_counts['Truck']}", (255, 255, 255), 0.48, 297),
        (f"Line total: {sum(line_counts.values())}", (0, 165, 255), 0.52, 322)
    ]

    for text, color, scale, y in lines:
        cv2.putText(frame, text, (22, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 2 if scale >= 0.52 else 1, cv2.LINE_AA)

    mode = "ON" if night_mode else "OFF"
    cv2.putText(frame, f"Low-light enhancement: {mode}", (20, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(frame, f"Processing FPS: {processing_fps:.1f}", (20, 395),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

# =====================================================
# 6. VÒNG LẶP CHÍNH (MAIN PROCESS)
# =====================================================

def run_vehicle_counting():
    source, night_mode, source_label = choose_source()
    if source is None:
        return
    if isinstance(source, str) and not os.path.isfile(source):
        print("\nERROR: Video file not found:", source)
        return

    print("\nLoading YOLO model...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as error:
        print("Could not load YOLO model:", error)
        return

    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) and os.name == "nt" else cv2.VideoCapture(source)
    if not cap.isOpened():
        print("ERROR: Cannot open video or webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 25.0 if fps <= 1 or fps > 120 else fps

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"vehicle_result_{source_label}_{timestamp}.mp4"

    writer, roi, line_position = None, None, None
    roi_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
    line_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
    counted_roi_ids, counted_line_ids = set(), set()
    previous_positions = {}
    processing_fps = 0.0

    print("\nSystem started. Press Q or ESC to stop.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            start_time = cv2.getTickCount()
            height, width = frame.shape[:2]

            if roi is None:
                roi = build_roi(width, height)
                line_position = get_line_position(width, height)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            # Xử lý ảnh nếu là chế độ ban đêm
            detection_frame = enhance_night_image(frame) if night_mode else frame
            confidence = NIGHT_CONFIDENCE if night_mode else DAY_CONFIDENCE

            # Chạy Tracking
            results = model.track(
                detection_frame, persist=True, tracker="bytetrack.yaml",
                classes=VEHICLE_CLASSES, conf=confidence, iou=0.5,
                imgsz=IMAGE_SIZE, verbose=False
            )

            boxes = results[0].boxes
            draw_road_region(frame, roi, line_position)

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i].astype(int)
                    class_id = int(class_ids[i])
                    conf_val = float(confidences[i])
                    vehicle_name = VEHICLE_NAMES.get(class_id)
                    
                    if vehicle_name is None: continue

                    # Lấy tọa độ trung tâm mép dưới (bánh xe chạm đường)
                    point_x = int((x1 + x2) / 2)
                    point_y = int(y2)
                    inside_roi = is_inside_roi(point_x, point_y, roi)
                    track_id = int(track_ids[i]) if track_ids is not None else None

                    color = (0, 255, 0) if inside_roi else (255, 165, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (point_x, point_y), 5, (0, 0, 255), -1)
                    
                    label = f"{vehicle_name} ID:{track_id if track_id else 'NA'} {conf_val:.2f}"
                    cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

                    if track_id is None: continue

                    # Đếm khi vào ROI
                    if inside_roi and track_id not in counted_roi_ids:
                        roi_counts[vehicle_name] += 1
                        counted_roi_ids.add(track_id)

                    # Đếm khi cắt ngang vạch Line
                    previous = previous_positions.get(track_id)
                    if previous is not None:
                        old_x, old_y = previous
                        if track_id not in counted_line_ids:
                            crossed = crossed_counting_line(old_x, old_y, point_x, point_y, line_position)
                            
                            if LINE_ORIENTATION == "horizontal":
                                cross_x, cross_y = (old_x + point_x) // 2, line_position
                                movement = abs(point_y - old_y)
                                max_jump = int(height * MAX_JUMP_REL)
                            else:
                                cross_x, cross_y = line_position, (old_y + point_y) // 2
                                movement = abs(point_x - old_x)
                                max_jump = int(width * MAX_JUMP_REL)

                            crossing_inside = is_inside_roi(cross_x, cross_y, roi)
                            reasonable_movement = movement <= max_jump

                            if crossed and crossing_inside and reasonable_movement:
                                line_counts[vehicle_name] += 1
                                counted_line_ids.add(track_id)

                    previous_positions[track_id] = (point_x, point_y)

            elapsed_seconds = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed_seconds > 0: processing_fps = 1.0 / elapsed_seconds

            draw_statistics(frame, roi_counts, line_counts, processing_fps, night_mode)
            if writer is not None: writer.write(frame)

            scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
            display_frame = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame

            cv2.imshow("YOLO Vehicle Counting", display_frame)
            if cv2.waitKey(max(1, int(1000 / fps)) if not isinstance(source, int) else 1) & 0xFF in (ord("q"), 27):
                break

    except Exception as error:
        print("\nERROR:", error)
    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_vehicle_counting()