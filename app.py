# Copyright 2026 Ivan Lobato / NeuralSoftX
# SPDX-License-Identifier: Apache-2.0
"""Streamlit web UI for interactive electron microscopy image restoration.

Author: Ivan Lobato
Email: ivan.lobato@neuralsoftx.com
"""
import time
import pathlib

import streamlit as st
import numpy as np

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="EM Image Restoration",
    page_icon="\U0001f52c",
    layout="wide",
)

from tk_r_em import load_network
from tk_r_em.file_io import load_image

SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "tif", "tiff", "ser", "dm3", "dm4"]


# --------- FIND ONNX MODELS ---------
def _get_available_models():
    """Return {model_stem: path} for every .onnx file in tk_r_em/models/."""
    models_dir = pathlib.Path(__file__).parent / "tk_r_em" / "models"
    try:
        return {f.stem: str(f) for f in sorted(models_dir.glob("*.onnx"))}
    except Exception:
        return {}


model_paths = _get_available_models()
model_names = list(model_paths.keys()) if model_paths else []

# --------- CSS ---------
st.markdown("""
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 0.8rem; }
div[data-testid="stSidebar"] .stButton > button {
    width: 100%; font-size: 1.1rem; font-weight: 600;
    background-color: #ff4b4b; color: white; border: none;
    padding: 0.6rem 1rem; border-radius: 0.5rem;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #e03e3e;
}
div[data-testid="stSidebar"] .stButton > button:disabled {
    background-color: #555; color: #999; cursor: not-allowed;
}
</style>
""", unsafe_allow_html=True)

# ============== SIDEBAR ==============
uploaded_file = st.sidebar.file_uploader(
    "Drag and drop EM image",
    type=SUPPORTED_FORMATS,
)

selected_model = None
model_path = None
if model_names:
    default_idx = model_names.index("sfr_hrstem") if "sfr_hrstem" in model_names else 0
    selected_model = st.sidebar.selectbox("Model", model_names, index=default_idx)
    model_path = model_paths[selected_model]
else:
    st.sidebar.warning("No .onnx models found in tk_r_em/models.")

inference_mode = st.sidebar.radio(
    "Inference mode",
    ["Whole image", "Patch-based"],
    index=0,
    help="Use patch-based for large images that do not fit in memory.",
)

patch_size = 256
stride = 128
if inference_mode == "Patch-based":
    patch_size = st.sidebar.slider("Patch size", 128, 512, value=256, step=32)
    stride = st.sidebar.slider("Stride", 64, 256, value=128, step=32)

# Restore button — disabled when no file loaded
run_restore = st.sidebar.button(
    "Restore", type="primary", use_container_width=True,
    disabled=(uploaded_file is None),
)

# --- Reset prediction when model changes ---
if "last_model" in st.session_state and st.session_state["last_model"] != selected_model:
    st.session_state["prediction"] = None
    st.session_state["inference_time"] = None

# --- Session State Init ---
_defaults = {
    "prediction": None,
    "last_uploaded_file": None,
    "last_model": None,
    "inference_time": None,
    "img_array": None,
    "flip_phase": 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============== HELPER ==============
def _normalise_to_uint8(img):
    """Normalise a 2D float array to [0, 255] uint8 RGB."""
    mn, mx = img.min(), img.max()
    denom = mx - mn if mx > mn else 1.0
    norm = np.clip((img - mn) / denom, 0.0, 1.0)
    return (np.stack([norm] * 3, axis=-1) * 255).astype(np.uint8)


# ============== MAIN LAYOUT ==============
left_pad, main, right_pad = st.columns([0.05, 0.9, 0.05])

with main:
    st.markdown(
        '<h1 style="text-align:center; margin:0.1rem 0 0.5rem 0;">'
        'EM Image Restoration Studio'
        '<span style="font-size:0.6em; margin-left:0.5rem; color:#888;">by</span> '
        '<span style="color:#ff4b4b;">Neuralsoftx</span></h1>',
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def _load_model(path):
        if path is None:
            return None
        return load_network(path)

    model = _load_model(model_path)

    # ================== IMAGE LOADING & INFERENCE ==================
    if uploaded_file is not None and model is not None:
        uploaded_file.seek(0)

        # Load image (only when file changes)
        if st.session_state["last_uploaded_file"] != uploaded_file.name:
            try:
                img_array = load_image(uploaded_file)
                st.session_state["img_array"] = img_array
                st.session_state["prediction"] = None
                st.session_state["inference_time"] = None
            except Exception as e:
                st.error(f"Error reading image: {e}")
                st.stop()
        else:
            img_array = st.session_state["img_array"]

        if img_array is None:
            st.error("Failed to load image.")
            st.stop()

        # Run inference on button click
        if run_restore:
            with st.spinner("Running restoration..."):
                t0 = time.time()
                if inference_mode == "Patch-based":
                    pred = model.predict_patch_based(
                        img_array, patch_size=patch_size, stride=stride, batch_size=16
                    )
                else:
                    pred = model.predict(img_array)
                t1 = time.time()

            st.session_state["prediction"] = pred
            st.session_state["last_uploaded_file"] = uploaded_file.name
            st.session_state["last_model"] = selected_model
            st.session_state["inference_time"] = t1 - t0
            st.session_state["flip_phase"] = 0
            st.rerun()

        prediction = st.session_state["prediction"]
        has_prediction = prediction is not None

        # ================== BUILD DISPLAY IMAGES ==================
        raw_uint8 = _normalise_to_uint8(img_array)

        if has_prediction:
            restored_uint8 = _normalise_to_uint8(prediction)

            # Info bar
            ms = st.session_state["inference_time"] * 1000
            mode_label = "patch-based" if inference_mode == "Patch-based" else "whole image"
            st.markdown(
                f'<div style="font-size:20px; font-weight:600; margin:0.1rem 0 0.5rem 0;">'
                f'Inference: <span style="color:#ff4b4b;">{ms:.1f} ms</span>'
                f'<span style="font-size:16px; margin-left:0.75rem;">'
                f'Model: <code>{selected_model}</code> | Mode: <code>{mode_label}</code>'
                f'</span></div>',
                unsafe_allow_html=True,
            )

            # Flip controls
            img_h, img_w = raw_uint8.shape[:2]

            c_flip_cb, c_flip_val = st.columns([0.15, 0.85])
            with c_flip_cb:
                flip_active = st.checkbox("Flip", key="flip_checkbox")
            with c_flip_val:
                st.text_input("Flip (s)", value=st.session_state.get("flip_time_text", "0.25"),
                              key="flip_time_text", label_visibility="collapsed")

            try:
                flip_time_val = float(st.session_state.get("flip_time_text", "0.25"))
                if flip_time_val <= 0:
                    flip_time_val = 0.25
            except (ValueError, TypeError):
                flip_time_val = 0.25

            # Display width — always visible
            auto_width = int(1562 * (img_w / img_h))
            display_width = st.slider("Display width (px)", 256, 3080,
                                      value=min(auto_width, 1200), step=50)

            if not flip_active:
                from streamlit_image_comparison import image_comparison
                image_comparison(
                    img1=raw_uint8,
                    img2=restored_uint8,
                    label1="Original",
                    label2="Restored",
                    starting_position=50,
                    show_labels=True,
                    make_responsive=False,
                    in_memory=True,
                    width=display_width,
                )

            else:
                @st.fragment
                def display_loop(img_res, img_raw, width_px, interval):
                    if "flip_phase" not in st.session_state:
                        st.session_state["flip_phase"] = 0
                    st.session_state["flip_phase"] = 1 - st.session_state["flip_phase"]

                    if st.session_state["flip_phase"] == 1:
                        current_img, caption = img_raw, "Original"
                    else:
                        current_img, caption = img_res, "Restored"

                    st.image(current_img, width=width_px, caption=caption, clamp=False)
                    time.sleep(interval)
                    st.rerun()

                display_loop(restored_uint8, raw_uint8, display_width, flip_time_val)

        else:
            # No prediction yet — small centred preview
            st.caption(
                f"Image loaded: {img_array.shape[1]} x {img_array.shape[0]} — "
                f"select model and click **Restore**"
            )
            c1, c2, c3 = st.columns([1, 1, 1])
            with c2:
                st.image(raw_uint8, caption="Preview", width="stretch")

    else:
        st.info("Upload an EM image and click **Restore** to begin.")
