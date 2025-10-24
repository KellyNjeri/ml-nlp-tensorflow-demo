# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E2F;
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        color: #BB86FC;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
    }
    .prediction {
        color: #03DAC6;
        font-size: 1.5em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>MNIST Digit Classifier 🔢</div>", unsafe_allow_html=True)
st.write("Upload a **handwritten digit (0–9)** and let the model predict it for you!")

# Load model
import os
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "mnist_model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📂 Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    
    # Preprocess
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
    
    with col2:
        st.markdown(f"<p class='prediction'>Predicted Digit: {predicted_digit}</p>", unsafe_allow_html=True)
        st.progress(float(confidence))
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # Bar chart
        fig, ax = plt.subplots(facecolor="#1E1E2F")
        ax.bar(range(10), prediction[0], color="#BB86FC")
        ax.set_xlabel('Digits', color='white')
        ax.set_ylabel('Probability', color='white')
        ax.set_title('Prediction Probabilities', color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
