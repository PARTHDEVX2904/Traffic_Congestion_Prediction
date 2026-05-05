import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Traffic Congestion Predictor",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Traffic Congestion Predictor")
st.markdown("Upload a traffic image and get real-time congestion prediction.")

# Sidebar
st.sidebar.header("📍 Location Settings")
lat = st.sidebar.number_input("Latitude",  value=12.9716, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=77.5946, format="%.4f")
st.sidebar.markdown("---")
st.sidebar.markdown("**Default:** Bengaluru, India")
st.sidebar.markdown("Change to your junction's coordinates.")

# Upload
uploaded_file = st.file_uploader("Upload a traffic image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Input Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    if st.button("🔍 Predict Congestion", type="primary"):
        with st.spinner("Analysing..."):
            uploaded_file.seek(0)
            response = requests.post(
                "http://localhost:8000/predict",
                files={"file": ("image.jpg", uploaded_file, "image/jpeg")},
                data={"lat": lat, "lon": lon}
            )

        if response.status_code == 200:
            data = response.json()

            with col2:
                st.subheader("🎯 Prediction Result")

                level = data["congestion"]
                confidence = data["confidence"]
                color = data["color"]

                # Congestion badge
                if color == "green":
                    st.success(f"### ✅ {level}")
                elif color == "orange":
                    st.warning(f"### ⚠️ {level}")
                else:
                    st.error(f"### 🚨 {level}")

                st.metric("Confidence", f"{confidence}%")
                st.markdown("---")

                # Vehicle counts
                st.subheader("🚗 Vehicle Detection")
                counts = data["vehicle_counts"]
                vcol1, vcol2, vcol3, vcol4 = st.columns(4)
                vcol1.metric("Cars",        counts["car"])
                vcol2.metric("Motorcycles", counts["motorcycle"])
                vcol3.metric("Buses",       counts["bus"])
                vcol4.metric("Trucks",      counts["truck"])

                st.metric("Total Vehicles", data["total_vehicles"])
                st.metric("Density Score",  data["density"])
                st.markdown("---")

                # Weather
                st.subheader("🌦️ Weather Conditions")
                weather = data["weather"]
                wcol1, wcol2 = st.columns(2)
                wcol1.metric("Visibility",  f"{weather['visibility_m']} m")
                wcol1.metric("Humidity",    f"{weather['humidity']}%")
                wcol2.metric("Rain",        f"{weather['rain_mm']} mm/hr")
                wcol2.metric("Temperature", f"{weather['temp']}°C")
                st.caption(f"Condition: {weather['description'].title()}")

        else:
            st.error(f"API error: {response.text}")