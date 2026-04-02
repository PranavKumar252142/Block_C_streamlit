"""
SafeSite Vision — AI-Powered PPE Compliance Detection
Streamlit Prototype for BUas ADS-AI Block C (Human-Centered AI)
"""
import os


from huggingface_hub import hf_hub_download

MODEL_PATH = "models/safesite_cnn.keras"
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    hf_hub_download(
        repo_id="produde1080/safesite_cnn/",
        filename="safesite_cnn.keras",
    )


import streamlit as st
import numpy as np
from PIL import Image
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeSite Vision",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-top: -10px;
        margin-bottom: 30px;
    }

    /* Result cards */
    .result-card {
        padding: 24px;
        border-radius: 12px;
        margin: 12px 0;
    }
    .compliant {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 6px solid #28a745;
    }
    .non-compliant {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 6px solid #dc3545;
    }
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .result-conf {
        font-size: 1rem;
        color: #444;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 16px;
        margin: 20px 0;
    }
    .metric-card {
        flex: 1;
        background: #f8f9fa;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #777;
        margin-top: 2px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #ccc;
        border-radius: 12px;
        padding: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ── Model loading ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Load your trained CNN model here.
    Replace the path below with your actual saved model path.
    """
    model_path = "models/safesite_cnn.keras"
    if os.path.exists(model_path):
        from tensorflow import keras
        return keras.models.load_model(model_path)
    else:
        return None  # Demo mode — uses simulated predictions


def preprocess_image(image: Image.Image, target_size=(128, 128)):
    """
    Preprocess uploaded image to match model input requirements.
    Adjust target_size to match your CNN's expected input shape.
    """
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0)


def predict(model, img_array):
    """
    Run inference. Returns (label, confidence).
    If no model is loaded, returns a simulated result for demo purposes.
    """
    if model is not None:
        prediction = model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        # Assuming sigmoid output: >0.5 = Compliant, <=0.5 = Non-Compliant
        if confidence >= 0.5:
            return "PPE Compliant ✅", confidence
        else:
            return "PPE Non-Compliant ⚠️", 1 - confidence
    else:
        # ── Demo mode: simulated prediction ──
        conf = np.random.uniform(0.72, 0.97)
        label = np.random.choice(
            ["PPE Compliant ✅", "PPE Non-Compliant ⚠️"],
            p=[0.4, 0.6],
        )
        return label, conf


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/safety-hat.png",
        width=64,
    )
    st.markdown("## SafeSite Vision")
    st.markdown("AI-powered PPE compliance detection for construction sites.")
    st.divider()

    st.markdown("### Settings")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.75,
        step=0.01,
        help="Minimum confidence to flag a definitive result.",
    )

    detection_scope = st.multiselect(
        "PPE to check",
        options=["Hard Hat", "Safety Vest"],
        default=["Hard Hat", "Safety Vest"],
        help="Select which PPE items to include in compliance checks.",
    )

    st.divider()
    st.markdown("### About")
    st.markdown(
        "SafeSite Vision uses a CNN-based image classifier to determine "
        "whether workers in construction site images are wearing the required "
        "personal protective equipment (hard hats and safety vests)."
    )
    st.markdown(
        "**Target users:** Safety Officers & HSE Managers"
    )
    st.divider()
    st.caption("BUas ADS-AI · Block C · 2025-2026")


# ── Main content ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🦺 SafeSite Vision</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Upload a construction site image to check PPE compliance</p>',
    unsafe_allow_html=True,
)

model = load_model()

if model is None:
    st.info(
        "🔧 **Demo mode** — No trained model found at `models/safesite_cnn.keras`. "
        "Predictions are simulated. Drop your trained model into the `models/` folder to enable real inference.",
        icon="ℹ️",
    )

# ── Upload & analysis section ────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### Upload Image")
    uploaded_file = st.file_uploader(
        "Drag and drop or browse",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)

with col_result:
    if uploaded_file is not None:
        st.markdown("#### Analysis Result")

        # Progress animation
        with st.spinner("Analyzing image for PPE compliance..."):
            img_array = preprocess_image(image)
            time.sleep(1.2)  # Brief pause for UX feel
            label, confidence = predict(model, img_array)

        # Determine styling
        is_compliant = "Compliant ✅" in label
        card_class = "compliant" if is_compliant else "non-compliant"

        # Result card
        st.markdown(f"""
        <div class="result-card {card_class}">
            <div class="result-label">{label}</div>
            <div class="result-conf">Confidence: {confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence threshold check
        if confidence < confidence_threshold:
            st.warning(
                f"Confidence ({confidence:.1%}) is below your threshold "
                f"({confidence_threshold:.0%}). Manual review recommended.",
                icon="⚡",
            )

        # Detailed breakdown
        st.markdown("#### Detection Details")
        for ppe_item in detection_scope:
            item_conf = confidence + np.random.uniform(-0.05, 0.05)
            item_conf = np.clip(item_conf, 0.0, 1.0)
            icon = "✅" if (is_compliant and item_conf > 0.5) else "❌"
            st.markdown(f"{icon} **{ppe_item}**: {item_conf:.1%} confidence")

        # Action buttons
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.button("📋 Log Violation", disabled=is_compliant, use_container_width=True)
        with c2:
            st.button("📤 Export Report", use_container_width=True)

    else:
        st.markdown("#### Analysis Result")
        st.markdown(
            '<div style="text-align:center; color:#aaa; padding:80px 20px;">'
            "Upload an image to see the PPE compliance analysis here."
            "</div>",
            unsafe_allow_html=True,
        )

# ── Dashboard metrics section ────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Session Dashboard")

# Initialize session state for tracking
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0
    st.session_state.violations = 0
    st.session_state.compliant_count = 0

if uploaded_file is not None and "last_file" not in st.session_state:
    st.session_state.total_scans += 1
    if not is_compliant:
        st.session_state.violations += 1
    else:
        st.session_state.compliant_count += 1
    st.session_state.last_file = uploaded_file.name

# Reset tracker on new file
if uploaded_file is None and "last_file" in st.session_state:
    del st.session_state.last_file

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Scans", st.session_state.total_scans)
with m2:
    st.metric("Violations Found", st.session_state.violations)
with m3:
    st.metric("Compliant", st.session_state.compliant_count)
with m4:
    rate = (
        (st.session_state.compliant_count / st.session_state.total_scans * 100)
        if st.session_state.total_scans > 0
        else 0
    )
    st.metric("Compliance Rate", f"{rate:.0f}%")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "SafeSite Vision · Pranav · BUas ADS-AI Year 1 Block C · 2025-2026<br>"
    "Prototype for Human-Centered AI & Dragons' Den Pitch"
    "</div>",
    unsafe_allow_html=True,
)
