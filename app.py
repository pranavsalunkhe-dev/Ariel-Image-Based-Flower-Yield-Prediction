import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide", page_title="Flower Yield MVP")

st.title("🌸 Flower-Yield MVP")
st.markdown("""
Upload an **aerial/top-down image** of a flower field (or use the sample).  
The app uses **simple color segmentation** to count flower blobs and converts  
count → yield (kg) → income using your calibration values.
""")

# ---------- SIDEBAR SETTINGS ----------
with st.sidebar:
    st.header("⚙️ Settings & Calibration")

    mass_per_flower_g = st.number_input(
        "Average mass per flower (grams)", 
        min_value=0.1, value=0.6, step=0.1,
        help="Weigh ~30 flowers and use the mean (g)."
    )
    price_per_kg = st.number_input(
        "Price per kg (₹ or local currency)", 
        min_value=0.0, value=200.0, step=1.0
    )
    area_m2 = st.number_input(
        "Area of the image (m²) — optional", 
        min_value=0.0, value=0.0, step=1.0, 
        help="Enter measured area in sq.m if known. Helps estimate per area."
    )

    st.markdown("### 🎨 Color Threshold (HSV)")
    h_min = st.slider("Hue min", 0, 179, 5)
    h_max = st.slider("Hue max", 0, 179, 25)
    s_min = st.slider("Sat min", 0, 255, 80)
    s_max = st.slider("Sat max", 0, 255, 255)
    v_min = st.slider("Val min", 0, 255, 100)
    v_max = st.slider("Val max", 0, 255, 255)

    st.markdown("### 📏 Detection Area Filters")
    min_area = st.slider("Min blob area (px)", 1, 2000, 50)
    max_area = st.slider("Max blob area (px)", 1, 1000, 25)

# ---------- MAIN AREA ----------
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload aerial image (jpg/png). If unsure, use sample.", 
        type=["jpg", "jpeg", "png"]
    )
    use_sample = st.button("Use Sample Image")

# ---------- IMAGE HANDLING ----------
if uploaded_file is None and not use_sample:
    st.info("Upload an image or press **'Use Sample Image'** to try the demo.")
    st.stop()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
else:
    # Change this path to your actual sample image location
    sample_path = r"C:\MiniProject\mini\images\R.jpeg"
    if not os.path.exists(sample_path):
        st.error(f"Sample image not found at: {sample_path}")
        st.stop()
    image = Image.open(sample_path).convert("RGB")


# ---------- IMAGE PROCESSING ----------
img_array = np.array(image)
display_img = img_array.copy()

# Convert to HSV and threshold
hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
lower = np.array([h_min, s_min, v_min])
upper = np.array([h_max, s_max, v_max])
mask = cv2.inRange(hsv, lower, upper)

# Morphological cleaning
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

# Find contours (detected blobs)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]

# ---------- CALCULATIONS ----------
count = len(filtered)
yield_kg = (count * mass_per_flower_g) / 1000.0
income = yield_kg * price_per_kg

# ---------- DRAW RESULTS ----------
out = display_img.copy()
for cnt in filtered:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)
st.markdown("### 📷 Input Image")
st.image(image,  use_container_width=True)

st.markdown("### 🟢 Detection Mask (HSV Threshold)")
st.image(mask, use_container_width=True, clamp=True)

# ---------- OUTPUT & DOWNLOAD ----------
out_pil = Image.fromarray(out)
st.image(out_pil, caption="Detections Overlay",  use_container_width=True)

buf = io.BytesIO()
out_pil.save(buf, format="JPEG")
buf.seek(0)
st.download_button(
    "📥 Download Overlay Image", 
    data=buf, 
    file_name="overlay.jpg", 
    mime="image/jpeg"
)


st.markdown("### 📊 Results")
st.metric("Detected flower blobs", f"{count}")
st.metric("Estimated yield (kg)", f"{yield_kg:.2f}")
st.metric("Estimated income ", f"{income:.0f} ₹")
# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
**Capture Tips (for better results):**
- Take images near midday with even lighting.
- Keep camera directly above the field (top-down).
- Maintain consistent height.
- Add a meter stick in one photo for area calibration.
""")

