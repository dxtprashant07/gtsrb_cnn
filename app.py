import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pandas as pd

# Traffic sign class names (GTSRB dataset)
CLASS_NAMES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Page configuration
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button { width: 100%; background-color: #FF4B4B; color: white; font-weight: bold; }
    .prediction-box {
        padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load trained model from file"""
    try:
        model = keras.models.load_model('gtsrb_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, img_size=(48, 48)):
    """
    Preprocess input image:
    - Convert to RGB
    - Resize using PIL (no OpenCV)
    - Normalize to [0,1]
    - Expand dimensions to batch format
    """
    image = image.convert("RGB")
    image = image.resize(img_size)
    img_array = np.array(image).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_sign(model, image):
    """Make prediction using the trained CNN"""
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)

    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    top5_indices = np.argsort(predictions[0])[-5:][::-1]
    top5_classes = [CLASS_NAMES[i] for i in top5_indices]
    top5_probs = [predictions[0][i] for i in top5_indices]

    return predicted_class, confidence, top5_classes, top5_probs

# --- Streamlit UI ---
def main():
    st.title("🚦 Traffic Sign Recognition System")
    st.markdown("### Deep Learning powered traffic sign classifier")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
            Deep Learning model trained on GTSRB dataset.
            - 43 traffic sign classes
            - Real-time prediction
            - Top-5 confidence scores
        """)
        st.header("Model Info")
        st.metric("Total Classes", "43")
        st.metric("Input Size", "48×48")
        st.metric("Model Type", "CNN")

    # Load model
    model = load_model()
    if model is None:
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload or Capture Image")

        uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        camera_capture = st.camera_input("Or capture using camera")

        image_source = uploaded if uploaded else camera_capture

        if image_source:
            image = Image.open(image_source)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Classify Traffic Sign"):
                with st.spinner("Processing..."):
                    pred_class, conf, top5_classes, top5_probs = predict_sign(model, image)

                    st.session_state['pred_class'] = pred_class
                    st.session_state['conf'] = conf
                    st.session_state['top5_classes'] = top5_classes
                    st.session_state['top5_probs'] = top5_probs

    with col2:
        st.header("📊 Prediction")

        if "pred_class" in st.session_state:
            pred_class = st.session_state["pred_class"]
            confidence = st.session_state["conf"]
            top5_classes = st.session_state["top5_classes"]
            top5_probs = st.session_state["top5_probs"]

            st.subheader("🎯 Predicted Sign")
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="color:#FF4B4B;">{CLASS_NAMES[pred_class]}</h2>
                <h3>Confidence: {confidence*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(confidence))

            # Top-5
            st.subheader("📈 Top-5 Predictions")
            df = pd.DataFrame({
                "Traffic Sign": top5_classes,
                "Probability (%)": [f"{p*100:.2f}" for p in top5_probs]
            })
            st.dataframe(df)

            st.bar_chart(pd.DataFrame(top5_probs, index=top5_classes, columns=["Confidence"]))

        else:
            st.info("📌 Upload an image to see predictions")

if __name__ == "__main__":
    main()
