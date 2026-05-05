from datetime import datetime

def build_feature_vector(counts, total, density, weather):
    """
    Combines vehicle detection + weather + temporal signals
    into a flat feature vector for the classifier.
    """
    now = datetime.now()
    hour = now.hour
    day = now.weekday()  # 0=Monday, 6=Sunday

    is_peak = 1 if (7 <= hour <= 10) or (17 <= hour <= 20) else 0
    is_weekend = 1 if day >= 5 else 0

    truck_ratio = round(counts["truck"] / total, 4) if total > 0 else 0.0
    moto_ratio  = round(counts["motorcycle"] / total, 4) if total > 0 else 0.0
    bus_ratio   = round(counts["bus"] / total, 4) if total > 0 else 0.0

    features = [
        total,                          # 0 - total vehicle count
        density,                        # 1 - density score
        truck_ratio,                    # 2 - truck ratio
        moto_ratio,                     # 3 - motorcycle ratio
        bus_ratio,                      # 4 - bus ratio
        weather["visibility_m"],        # 5 - visibility in metres
        weather["humidity"],            # 6 - humidity %
        weather["rain_mm"],             # 7 - rain mm/hr
        weather["temp"],                # 8 - temperature
        weather["is_rain"],             # 9 - rain flag
        weather["is_fog"],              # 10 - fog flag
        weather["is_night"],            # 11 - night flag
        hour,                           # 12 - hour of day
        is_peak,                        # 13 - peak hour flag
        is_weekend,                     # 14 - weekend flag
    ]

    return features


FEATURE_NAMES = [
    "total_vehicles", "density", "truck_ratio", "moto_ratio", "bus_ratio",
    "visibility_m", "humidity", "rain_mm", "temp",
    "is_rain", "is_fog", "is_night",
    "hour", "is_peak", "is_weekend"
]


if __name__ == "__main__":
    # Test with dummy data
    counts = {"car": 3, "motorcycle": 0, "bus": 2, "truck": 0}
    total = 5
    density = 0.1474
    weather = {
        "visibility_m": 10000, "humidity": 50, "rain_mm": 0.0,
        "temp": 25.0, "is_rain": 0, "is_fog": 0, "is_night": 0
    }

    features = build_feature_vector(counts, total, density, weather)
    print("Feature vector:")
    for name, val in zip(FEATURE_NAMES, features):
        print(f"  {name}: {val}")