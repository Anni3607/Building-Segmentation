
"""
Simple Streamlit app to run building footprint segmentation using the trained Keras model.
Save this string to app.py (the notebook also writes app.py to disk).
"""

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tempfile
from PIL import Image

st.set_page_config(page_title="Building Footprint Segmentation", layout="centered")

st.title("Building Footprint Segmentation")
st.markdown("Upload an aerial image (RGB). The app runs the trained model and shows predicted building masks.")

# Path to model file (ensure in same directory or provide path)
MODEL_PATH = "inria_unet_final.h5"

@st.cache_resource
def load_seg_model(path):
    try:
        model = load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model at {path}: {e}")
        return None

model = load_seg_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload an RGB image (jpg/png/tif)", type=['jpg', 'jpeg', 'png', 'tif', 'tiff'])
if uploaded_file is not None and model is not None:
    # Read uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    h_orig, w_orig = img_np.shape[:2]

    # Resize to model input size (assumes 256x256, change if different)
    input_size = (256, 256)
    img_resized = cv2.resize(img_np, input_size, interpolation=cv2.INTER_LINEAR)
    x = img_resized.astype('float32') / 255.0
    x = np.expand_dims(x, 0)

    with st.spinner("Running inference..."):
        pred = model.predict(x)[0]
        pred_bin = (pred > 0.5).astype('uint8') * 255
        pred_up = cv2.resize(pred_bin.squeeze(), (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    st.subheader("Input Image")
    st.image(img_np, use_column_width=True)

    st.subheader("Predicted Mask (binary)")
    st.image(pred_up, clamp=True, use_column_width=True)

    # Overlay
    overlay = img_np.copy()
    red_mask = np.zeros_like(overlay)
    red_mask[:,:,0] = pred_up
    alpha = 0.5
    overlayed = cv2.addWeighted(overlay, 1.0, red_mask, alpha, 0)

    st.subheader("Overlay")
    st.image(overlayed, use_column_width=True)

    # Download predicted mask
    import io
    from PIL import Image as PILImage
    mask_pil = PILImage.fromarray(pred_up.astype('uint8'))
    buf = io.BytesIO()
    mask_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download predicted mask (PNG)", data=byte_im, file_name="pred_mask.png", mime="image/png")
else:
    if model is None:
        st.warning("Model not found. Make sure 'inria_unet_final.h5' exists in this directory.")
    st.info("Upload an image to run the model.")
