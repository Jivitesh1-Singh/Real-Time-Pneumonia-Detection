import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("pneumonia_cnn_model.h5")

IMG_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA"]

st.title("Chest X-Ray Pneumonia Detection")
st.write("Upload a chest X-ray image to get real-time prediction.")

uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Uploaded X-ray Image", use_column_width=True)

    # Preprocess
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Prediction
    prediction = model.predict(img)
    result = CLASSES[int(prediction > 0.5)]

    st.subheader("Prediction Result:")
    st.success(result)
