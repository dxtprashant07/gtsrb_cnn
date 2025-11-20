import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import pandas as pd

# ------------- CLASS NAMES FOR 43 CLASSES ---------------
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

# -------------------------------------------------------
#                STREAMLIT PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
#            LOAD MODEL SAFELY (CACHED)
# -------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model("gtsrb_cnn_model.h5")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


model = load_model()


# -------------------------------------------------------
#     PREPROCESS IMAGE (AUTO-DETECT MODEL INPUT SIZE)
# -------------------------------------------------------
def preprocess_image(image):
    """Resize image based on model input shape."""

    # Automatically detect model required input size
    input_shape = model.input_shape  # (None, H, W, 3)
    img_size = (input_shape[1], input_shape[2])

    img_array = np.array(image)

    # Resize exactly to model size
    img_resized = cv2.resize(img_array, img_size)

    # Normalize
    img_normalized = img_resized.astype('float32') / 255.0

    # Add batch dim
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch


# -------------------------------------------------------
#                  PREDICTION FUNCTION
# -------------------------------------------------------
def predict_sign(model, image):
    processed_img = preprocess_image(image)

    predictions = model.predict(processed_img, verbose=0)

    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5_classes = [CLASS_NAMES[i] for i in top_5_idx]
    top_5_probs = [predictions[0][i] for i in top_5_idx]

    return predicted_class, confidence, top_5_classes, top_5_probs


# -------------------------------------------------------
#                  STREAMLIT MAIN APP
# -------------------------------------------------------
def main():
    st.title("🚦 Traffic Sign Recognition System")
    st.markdown("### Deep Learning-powered traffic sign classifier")
    st.markdown("---")

    if model is None:
        st.error("❌ Please train and place gtsrb_cnn_model.h5 in the folder.")
        return

    col1, col2 = st.columns([1, 1])

    # ----------- LEFT COLUMN: UPLOAD IMAGE ------------
    with col1:
        st.header("📤 Upload or Capture Image")

        uploaded_file = st.file_uploader("Upload Traffic Sign Image", 
                                         type=['png', 'jpg', 'jpeg', 'webp'])

        camera_input = st.camera_input("Or capture using camera")

        image_source = uploaded_file if uploaded_file else camera_input

        if image_source:
            image = Image.open(image_source)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Classify Traffic Sign"):
                with st.spinner("Predicting..."):
                    pred_class, conf, top5_classes, top5_probs = predict_sign(model, image)

                    st.session_state["prediction"] = {
                        "pred_class": pred_class,
                        "confidence": conf,
                        "top5_classes": top5_classes,
                        "top5_probs": top5_probs
                    }

    # ------------- RIGHT COLUMN: RESULTS -----------------
    with col2:
        st.header("📊 Prediction Results")

        if "prediction" in st.session_state:
            pred = st.session_state["prediction"]

            st.markdown("### 🎯 Predicted Sign")
            st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color:#FF4B4B">{CLASS_NAMES[pred['pred_class']]}</h2>
                    <h4>Confidence: {pred['confidence']*100:.2f}%</h4>
                </div>
            """, unsafe_allow_html=True)

            st.progress(float(pred['confidence']))

            # Top 5 predictions
            st.markdown("### 📈 Top 5 Predictions")

            df = pd.DataFrame({
                "Traffic Sign": pred["top5_classes"],
                "Probability": [f"{p*100:.2f}%" for p in pred["top5_probs"]]
            })

            st.dataframe(df, use_container_width=True)

            st.bar_chart(
                pd.DataFrame({"Probability": pred["top5_probs"]},
                             index=pred["top5_classes"])
            )

        else:
            st.info("👆 Upload an image to get predictions.")

    st.markdown("---")
    st.markdown("### ✔ Works with ANY image size — auto-resizes to model input")


if __name__ == "__main__":
    main()
