from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def detect_vehicles(image_input):
    results = model(image_input, verbose=False)[0]

    counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    box_areas = []
    img_area = results.orig_shape[0] * results.orig_shape[1]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            label = VEHICLE_CLASSES[cls_id]
            counts[label] += 1
            x1, y1, x2, y2 = box.xyxy[0]
            box_areas.append(float((x2 - x1) * (y2 - y1)))

    total = sum(counts.values())
    density = round(float(sum(box_areas)) / float(img_area), 4) if box_areas else 0.0
    annotated = results.plot()

    return counts, total, density, annotated


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python model/detector.py <image_path>")
    else:
        counts, total, density, _ = detect_vehicles(path)
        print(f"Counts : {counts}")
        print(f"Total  : {total}")
        print(f"Density: {density}")