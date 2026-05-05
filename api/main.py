from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np
import sys
import os
from PIL import Image
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.detector import detect_vehicles
from model.weather import get_weather
from model.features import build_feature_vector
from model.classifier import predict_congestion

app = FastAPI(title="Traffic Congestion Predictor")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: float = Form(default=12.9716),
    lon: float = Form(default=77.5946)
):
    try:
        # Step 1 — read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(image)

        # Step 2 — detect vehicles
        counts, total, density, _ = detect_vehicles(img_array)

        # Step 3 — get weather
        weather = get_weather(lat, lon)

        # Step 4 — build features
        features = build_feature_vector(counts, total, density, weather)

        # Step 5 — predict
        result = predict_congestion(features)

        return JSONResponse({
            "congestion": result["level"],
            "confidence": result["confidence"],
            "color": result["color"],
            "vehicle_counts": counts,
            "total_vehicles": total,
            "density": density,
            "weather": {
                "description": weather["description"],
                "visibility_m": weather["visibility_m"],
                "humidity": weather["humidity"],
                "rain_mm": weather["rain_mm"],
                "temp": weather["temp"]
            }
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)