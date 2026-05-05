import requests

API_KEY = "295b8a18d313968d31814132f19abc68" 

def get_weather(lat, lon):
    """
    Fetch weather for a given lat/lon.
    Returns a dict of weather features.
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        weather_id = data["weather"][0]["id"]
        visibility = data.get("visibility", 10000)
        humidity = data["main"]["humidity"]
        rain_mm = data.get("rain", {}).get("1h", 0.0)
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]

        # Derived flags
        is_rain = 1 if 500 <= weather_id <= 531 else 0
        is_fog  = 1 if 700 <= weather_id <= 771 else 0
        is_night = 1 if weather_id == 800 and data.get("dt", 0) > data.get("sys", {}).get("sunset", 0) else 0

        return {
            "visibility_m": visibility,
            "humidity": humidity,
            "rain_mm": rain_mm,
            "temp": temp,
            "is_rain": is_rain,
            "is_fog": is_fog,
            "is_night": is_night,
            "description": description
        }

    except Exception as e:
        print(f"Weather API failed: {e}, using defaults")
        return {
            "visibility_m": 10000,
            "humidity": 50,
            "rain_mm": 0.0,
            "temp": 25.0,
            "is_rain": 0,
            "is_fog": 0,
            "is_night": 0,
            "description": "clear"
        }


if __name__ == "__main__":
    # Test with Bengaluru coordinates
    url = f"http://api.openweathermap.org/data/2.5/weather?lat=12.9716&lon=77.5946&appid={API_KEY}&units=metric"
    response = requests.get(url, timeout=5)
    print(response.json())
    # for k, v in weather.items():
    #     print(f"{k}: {v}")