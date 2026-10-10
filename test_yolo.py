from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Run detection
results = model("images/traffic.jpg")

# Count vehicles
vehicle_counts = {
    "Car": 0,
    "Motorcycle": 0,
    "Bus": 0,
    "Truck": 0
}

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])

        if class_id in vehicle_classes:
            vehicle_name = vehicle_classes[class_id]
            vehicle_counts[vehicle_name] += 1

# Display results
print("\n===== VEHICLE COUNT =====")

for vehicle, count in vehicle_counts.items():
    print(f"{vehicle}: {count}")

total = sum(vehicle_counts.values())

print("-------------------------")
print(f"Total vehicles: {total}")