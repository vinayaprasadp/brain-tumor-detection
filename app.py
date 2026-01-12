import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 150

# Load trained model
model = tf.keras.models.load_model("model/brain_tumor_cnn.h5")

st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠")

st.title("🧠 Brain Tumor Detection System")
st.write("Upload a brain MRI image to check for tumor presence.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess image
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]

    

if prediction < 0.3:
    st.success("✅ No Tumor Detected")
    st.write("Severity: None")
elif prediction < 0.6:
    st.warning("🟡 Tumor Detected")
    st.write("Severity: Mild")
elif prediction < 0.85:
    st.warning("🟠 Tumor Detected")
    st.write("Severity: Moderate")
else:
    st.error("🔴 Tumor Detected")
    st.write("Severity: Severe")

st.write(f"Confidence Score: {prediction:.2f}")

