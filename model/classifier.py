import pickle
import numpy as np

LABELS = {0: "Free Flow", 1: "Moderate", 2: "Congested"}
COLORS = {0: "green", 1: "orange", 2: "red"}

with open("congestion_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_congestion(feature_vector):
    """
    Takes a flat feature vector (list of 15 values).
    Returns label string, confidence, and color.
    """
    X = np.array(feature_vector).reshape(1, -1)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = round(float(proba[pred]) * 100, 1)

    return {
        "level": LABELS[pred],
        "confidence": confidence,
        "color": COLORS[pred],
        "class_id": pred
    }


if __name__ == "__main__":
    # Test: heavy traffic, rainy, peak hour
    test_vector = [
        35, 0.72, 0.1, 0.2, 0.05,   # vehicles + ratios
        800, 85, 4.5, 28.0,          # weather
        1, 0, 0,                     # flags
        8, 1, 0                      # time
    ]
    result = predict_congestion(test_vector)
    print(f"Level     : {result['level']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Color     : {result['color']}")