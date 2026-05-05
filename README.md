# Traffic Congestion Prediction

A real-time traffic congestion prediction system that combines computer vision (YOLOv8) and machine learning (XGBoost) to classify road congestion from images. Upload a photo of a road and get an instant prediction — **Free Flow**, **Moderate**, or **Congested** — enriched with live weather data.

---

## How It Works

```
Image Upload → YOLOv8 Vehicle Detection → Weather Fetch → Feature Engineering → XGBoost Classifier → Prediction
```

1. **Detector** (`model/detector.py`) — YOLOv8n counts cars, motorcycles, buses, and trucks; computes a density score (bounding-box area ÷ image area).
2. **Weather** (`model/weather.py`) — fetches current conditions (visibility, rain, wind) from OpenWeatherMap for the provided coordinates.
3. **Feature builder** (`model/features.py`) — assembles a fixed 15-feature vector: vehicle counts + weather + temporal signals (hour, peak-hour flag, weekend flag).
4. **Classifier** (`model/classifier.py`) — XGBoost model predicts one of three congestion levels with a confidence score.

---

## Project Structure

```
traffic-congestion/
├── api/
│   └── main.py              # FastAPI app — /health + /predict endpoints
├── model/
│   ├── detector.py          # YOLOv8 vehicle detection
│   ├── weather.py           # OpenWeatherMap integration
│   ├── features.py          # 15-feature vector builder
│   └── classifier.py        # XGBoost congestion classifier
├── ui/
│   └── app.py               # Streamlit web UI
├── train_classifier.py      # Training script (synthetic dataset)
├── congestion_model.pkl     # Trained XGBoost model
├── yolov8n.pt               # YOLOv8 Nano weights
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

### Option A — Local (with virtual environment)

```powershell
# 1. Create and activate venv
cd traffic-congestion
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates congestion_model.pkl)
python train_classifier.py

# 4. Start the API (port 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 5. In a separate terminal, start the UI (port 8501)
streamlit run ui/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Option B — Docker

```powershell
cd traffic-congestion
docker-compose up --build
```

- UI → [http://localhost:8501](http://localhost:8501)
- API → [http://localhost:8000](http://localhost:8000)

---

## API Reference

### `GET /health`
Returns `{"status": "ok"}` — liveness probe.

### `POST /predict`
Accepts a multipart form with:
| Field | Type | Description |
|-------|------|-------------|
| `file` | image | Road image (JPEG/PNG) |
| `lat` | float | Latitude (default: Bengaluru 12.9716) |
| `lon` | float | Longitude (default: Bengaluru 77.5946) |

**Example response:**
```json
{
  "congestion_level": "Moderate",
  "confidence": 0.82,
  "vehicle_counts": { "car": 5, "motorcycle": 3, "bus": 1, "truck": 0 },
  "density_score": 0.34,
  "weather": { "visibility": 8000, "rain_1h": 0.0, "wind_speed": 3.2 }
}
```

**cURL example:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test.jpeg" \
  -F "lat=12.9716" \
  -F "lon=77.5946"
```

---

## Model Details

The XGBoost classifier is trained on a **synthetic dataset** of 3,000 samples (1,000 per class) generated with domain rules:

| Class | Key signals |
|-------|-------------|
| Free Flow | Low vehicle count, off-peak hours, good visibility |
| Moderate | Medium vehicle count, shoulder hours |
| Congested | High vehicle count, peak hours (8–10 AM / 5–8 PM), low visibility or rain |

The feature vector is always exactly **15 values** in a fixed order (see `model/features.py:FEATURE_NAMES`). Changing the schema requires retraining.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Object detection | [Ultralytics YOLOv8n](https://github.com/ultralytics/ultralytics) |
| Classification | XGBoost |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Weather data | OpenWeatherMap REST API |
| Containerization | Docker + Docker Compose |

---

## Requirements

- Python 3.10+
- `congestion_model.pkl` must exist before starting the API (run `train_classifier.py` first)
- Working directory must be `traffic-congestion/` when running any module (the `.pkl` is loaded relative to CWD)
- An OpenWeatherMap API key is configured in `model/weather.py` (failures fall back to neutral defaults silently)
