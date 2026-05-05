import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import random

random.seed(42)
np.random.seed(42)

# ── Labels ──────────────────────────────────────────
# 0 = free_flow
# 1 = moderate
# 2 = congested

def generate_sample(label):
    """Generate one realistic feature vector for a given congestion label."""

    if label == 0:  # free flow
        total      = random.randint(0, 8)
        density    = random.uniform(0.0, 0.15)
        visibility = random.randint(5000, 10000)
        rain_mm    = 0.0
        is_rain    = 0
        is_fog     = 0
        hour       = random.choice([1,2,3,4,5,11,13,15,22,23])
        is_peak    = 0

    elif label == 1:  # moderate
        total      = random.randint(8, 20)
        density    = random.uniform(0.15, 0.40)
        visibility = random.randint(2000, 6000)
        rain_mm    = random.uniform(0.0, 2.0)
        is_rain    = random.choice([0, 1])
        is_fog     = 0
        hour       = random.choice([6,10,11,12,13,14,16,20,21])
        is_peak    = random.choice([0, 1])

    else:  # congested
        total      = random.randint(20, 60)
        density    = random.uniform(0.40, 0.95)
        visibility = random.randint(200, 3000)
        rain_mm    = random.uniform(1.0, 10.0)
        is_rain    = random.choice([0, 1])
        is_fog     = random.choice([0, 1])
        hour       = random.choice([7,8,9,17,18,19,20])
        is_peak    = 1

    truck_ratio = random.uniform(0.0, 0.3)
    moto_ratio  = random.uniform(0.0, 0.4)
    bus_ratio   = random.uniform(0.0, 0.2)
    humidity    = random.randint(30, 95)
    temp        = random.uniform(15.0, 42.0)
    is_night    = 1 if hour >= 20 or hour <= 5 else 0
    is_weekend  = random.choice([0, 1])

    return [
        total, density, truck_ratio, moto_ratio, bus_ratio,
        visibility, humidity, rain_mm, temp,
        is_rain, is_fog, is_night,
        hour, is_peak, is_weekend
    ]


def generate_dataset(n=3000):
    X, y = [], []
    per_class = n // 3
    for label in [0, 1, 2]:
        for _ in range(per_class):
            X.append(generate_sample(label))
            y.append(label)
    return np.array(X), np.array(y)


FEATURE_NAMES = [
    "total_vehicles", "density", "truck_ratio", "moto_ratio", "bus_ratio",
    "visibility_m", "humidity", "rain_mm", "temp",
    "is_rain", "is_fog", "is_night",
    "hour", "is_peak", "is_weekend"
]

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    X, y = generate_dataset(3000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    print("\nTraining XGBoost...")
    clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        #use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_pred, y_test,
          target_names=["free_flow", "moderate", "congested"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    with open("congestion_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("\nModel saved to congestion_model.pkl ✅")