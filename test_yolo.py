
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = "yolo11n.pt"
IMAGE_PATH = "images/traffic.jpg"
OUTPUT_DIR = Path("results")

CONFIDENCE_THRESHOLD = 0.25

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


# ==========================================
# 2. LOAD IMAGE AND MODEL
# ==========================================

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not Path(IMAGE_PATH).exists():
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError(f"Cannot read image: {IMAGE_PATH}")

height, width = image.shape[:2]

print(f"Image size: {width} x {height}")

model = YOLO(MODEL_PATH)


# ==========================================
# 3. DEFINE POLYGON ROI
# ==========================================

# Polygon coordinates are based on the current
# 640 x 512 traffic.jpg image.
#
# The polygon covers the lower roadway area.
# Change these points if you want another area.

roi_points = np.array([
    [0, 340],
    [width - 1, 340],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.int32)

# Ensure all points are within the image
roi_points[:, 0] = np.clip(
    roi_points[:, 0], 0, width - 1
)
roi_points[:, 1] = np.clip(
    roi_points[:, 1], 0, height - 1
)

roi_contour = roi_points.reshape((-1, 1, 2))


# ==========================================
# 4. RUN YOLO DETECTION
# ==========================================

results = model.predict(
    source=image,
    conf=CONFIDENCE_THRESHOLD,
    verbose=False
)

result = results[0]


# ==========================================
# 5. INITIALIZE COUNTERS
# ==========================================

vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

vehicles_in_roi = []


# ==========================================
# 6. FILTER VEHICLES INSIDE ROI
# ==========================================

for box in result.boxes:

    class_id = int(box.cls[0].item())
    confidence = float(box.conf[0].item())

    # Ignore people, umbrellas and other objects
    if class_id not in VEHICLE_CLASSES:
        continue

    x1, y1, x2, y2 = map(
        int, box.xyxy[0].tolist()
    )

    # Calculate the center of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    # Check whether the center is inside the polygon
    inside_roi = cv2.pointPolygonTest(
        roi_contour,
        (float(center_x), float(center_y)),
        False
    ) >= 0

    # Count only vehicles inside the ROI
    if not inside_roi:
        continue

    vehicle_name = VEHICLE_CLASSES[class_id]

    vehicle_counts[vehicle_name] += 1

    vehicles_in_roi.append({
        "name": vehicle_name,
        "confidence": confidence,
        "box": (x1, y1, x2, y2),
        "center": (center_x, center_y)
    })


# ==========================================
# 7. DRAW ROI POLYGON
# ==========================================

# Draw a transparent green overlay
overlay = image.copy()

cv2.fillPoly(
    overlay,
    [roi_contour],
    (0, 180, 0)
)

image = cv2.addWeighted(
    overlay,
    0.15,
    image,
    0.85,
    0
)

# Draw the polygon border
cv2.polylines(
    image,
    [roi_contour],
    isClosed=True,
    color=(0, 255, 255),
    thickness=3
)

# Add ROI label
label_x = 10
label_y = max(25, int(roi_points[:, 1].min()) - 10)

cv2.putText(
    image,
    "ROI - COUNTING AREA",
    (label_x, label_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (0, 255, 255),
    2,
    cv2.LINE_AA
)


# ==========================================
# 8. DRAW VEHICLE BOXES AND CONFIDENCE
# ==========================================

for vehicle in vehicles_in_roi:

    x1, y1, x2, y2 = vehicle["box"]
    name = vehicle["name"]
    confidence = vehicle["confidence"]
    center_x, center_y = vehicle["center"]

    label = f"{name} {confidence:.2f}"

    # Draw bounding box
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Mark the center used for ROI filtering
    cv2.circle(
        image,
        (center_x, center_y),
        4,
        (0, 0, 255),
        -1
    )

    # Label background
    (text_width, text_height), baseline = (
        cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )
    )

    label_top = max(0, y1 - text_height - baseline - 6)

    cv2.rectangle(
        image,
        (x1, label_top),
        (x1 + text_width + 8,
         label_top + text_height + baseline + 6),
        (0, 255, 0),
        -1
    )

    cv2.putText(
        image,
        label,
        (x1 + 4, label_top + text_height + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


# ==========================================
# 9. DRAW COUNT TABLE
# ==========================================

total_vehicles = sum(vehicle_counts.values())

panel_x = 10
panel_y = 45
panel_width = 235
row_height = 27
panel_height = 43 + row_height * 5

cv2.rectangle(
    image,
    (panel_x, panel_y),
    (panel_x + panel_width, panel_y + panel_height),
    (35, 35, 35),
    -1
)

cv2.rectangle(
    image,
    (panel_x, panel_y),
    (panel_x + panel_width, panel_y + panel_height),
    (0, 255, 0),
    2
)

cv2.putText(
    image,
    "ROI VEHICLE COUNT",
    (panel_x + 8, panel_y + 27),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 255, 255),
    2,
    cv2.LINE_AA
)

row_y = panel_y + 53

for name, count in vehicle_counts.items():

    cv2.putText(
        image,
        f"{name}: {count}",
        (panel_x + 8, row_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    row_y += row_height

cv2.putText(
    image,
    f"TOTAL: {total_vehicles}",
    (panel_x + 8, row_y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 255, 0),
    2,
    cv2.LINE_AA
)


# ==========================================
# 10. SAVE RESULT
# ==========================================

output_path = OUTPUT_DIR / "traffic_roi_result.jpg"

if not cv2.imwrite(str(output_path), image):
    raise RuntimeError("Could not save the result image.")


# ==========================================
# 11. PRINT RESULTS
# ==========================================

print("\n========== ROI VEHICLE COUNT ==========")

for name, count in vehicle_counts.items():
    print(f"{name}: {count}")

print("---------------------------------------")
print(f"Total vehicles inside ROI: {total_vehicles}")
print(f"Vehicles outside ROI are not counted.")
print(f"Saved result: {output_path.resolve()}")


# ==========================================
# 12. DISPLAY RESULT
# ==========================================

cv2.imshow("Vehicle Counting - Polygon ROI", image)

print("\nPress Q or ESC to close the window.")

while True:
    key = cv2.waitKey(0) & 0xFF

    if key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()