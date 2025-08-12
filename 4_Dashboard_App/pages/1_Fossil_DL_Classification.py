# =========================================================================
#  Fossil Classifier - FINAL Version 2.3
#  - Adds an interactive slider to control segmentation sensitivity.
#  - Adds a live preview of the image that will be fed to the model.
# =========================================================================

import streamlit as st
import plotly.express as px
import tensorflow as tf
import numpy as np
import json
import gc
import cv2
import pathlib
import os
from PIL import Image
from streamlit_extras.image_selector import image_selector
from typing import Tuple
import pandas as pd

# --- Scientific & Image Processing Imports ---
from skimage import filters, morphology, measure
from packaging.version import parse as parse_version

# ---------------------------------------------------------------------------
# 1. CORE MODEL LOGIC & DEFINITIONS
# ---------------------------------------------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"; os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
tf.keras.mixed_precision.set_global_policy("mixed_float16")
CANONICAL_SIZE = (384, 384)
from tensorflow.keras.applications.convnext import preprocess_input as cvx_prep
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eff_prep

if parse_version(tf.__version__) < parse_version("2.9"):
    from tensorflow.keras.utils import register_keras_serializable, serialize_keras_object, deserialize_keras_object
else:
    from tensorflow.keras.saving import register_keras_serializable, serialize_keras_object, deserialize_keras_object

@tf.autograph.experimental.do_not_convert
def crop_to_bbox(img):
    gray=tf.image.rgb_to_grayscale(img);coords=tf.cast(tf.where(tf.squeeze(tf.where(gray>0.05,1,0))),tf.int32)
    def _crop():
        ymin,ymax=tf.reduce_min(coords[:,0])-5,tf.reduce_max(coords[:,0])+5;xmin,xmax=tf.reduce_min(coords[:,1])-5,tf.reduce_max(coords[:,1])+5
        h,w=tf.shape(img,tf.int32)[0],tf.shape(img,tf.int32)[1];ymin,ymax=tf.clip_by_value(ymin,0,h),tf.clip_by_value(ymax,0,h);xmin,xmax=tf.clip_by_value(xmin,0,w),tf.clip_by_value(xmax,0,w)
        empty=tf.logical_or(tf.less_equal(ymax,ymin),tf.less_equal(xmax,xmin));return tf.cond(empty,lambda:img,lambda:img[ymin:ymax,xmin:xmax])
    crop=tf.cond(tf.shape(coords)[0]>0,_crop,lambda:img);return tf.cast(tf.image.resize_with_pad(crop,*CANONICAL_SIZE),img.dtype)

@register_keras_serializable(package="Fossil")
class PatchEnsemble(tf.keras.Model):
    def __init__(self,main,patch,weak_ids,**kw):super().__init__(**kw);self.main,self.patch,self.weak=main,patch,tf.constant(weak_ids,dtype=tf.int64);self.m_prep=tf.keras.layers.Lambda(cvx_prep,name="main_prep");self.p_prep=tf.keras.layers.Lambda(eff_prep,name="patch_prep")
    def call(self,x,training=False):pm=self.main(self.m_prep(x),training=False);pp=self.patch(self.p_prep(tf.image.resize(x,(224,224))),training=False);top1=tf.argmax(pm,-1,output_type=tf.int64);better=tf.reduce_max(pp,-1)>tf.reduce_max(pm,-1);weak=tf.reduce_any(top1[:,None]==self.weak[None,:],axis=1);mask=tf.logical_and(weak,better);mask_f=tf.cast(mask[:,None],pm.dtype);out=pm*(1-mask_f)+pp*mask_f;return out/tf.reduce_sum(out,-1,keepdims=True)
    def get_config(self):base_config=super().get_config();config={"main":serialize_keras_object(self.main),"patch":serialize_keras_object(self.patch),"weak_ids":self.weak.numpy().tolist()};return {**base_config,**config}
    @classmethod
    def from_config(cls,config):main_model=deserialize_keras_object(config.pop("main"));patch_model=deserialize_keras_object(config.pop("patch"));return cls(main=main_model,patch=patch_model,**config)

# MODIFIED: Accepts a threshold_adjustment parameter
def segment_image(img_gray, assume_bright_fossil=True, invert_output=False, threshold_adjustment=0.0) -> Tuple[np.ndarray, bool]:
    try:
        blurred=cv2.GaussianBlur(img_gray,(3,3),0)
        # Adjust the Otsu threshold using the slider value
        otsu_thresh = filters.threshold_otsu(blurred)
        adjusted_thresh = otsu_thresh + threshold_adjustment
        
        binary=blurred > adjusted_thresh if assume_bright_fossil else blurred < adjusted_thresh
        
        cleaned=morphology.remove_small_objects(binary,min_size=200);cleaned=morphology.remove_small_holes(cleaned,area_threshold=100);cleaned=morphology.binary_closing(cleaned,morphology.disk(2))
        labeled=measure.label(cleaned)
        if labeled.max()==0: return None, False
        main_fossil_label=max(measure.regionprops(labeled),key=lambda x:x.area).label;mask=(labeled==main_fossil_label).astype(np.uint8)*255
        segmented_gray=cv2.bitwise_and(img_gray,img_gray,mask=mask)
        if invert_output: non_black_mask = segmented_gray > 0; segmented_gray[non_black_mask] = 255 - segmented_gray[non_black_mask]
        return segmented_gray, True
    except Exception as e: st.error(f"Error during segmentation: {e}"); return None, False

# # ---------------------------------------------------------------------------
# # 2. APP CONFIGURATION & STYLING
# # ---------------------------------------------------------------------------
# st.set_page_config(page_title="Fossil AI | Classifier", page_icon="🦕", initial_sidebar_state="expanded")
# ---------------------------------------------------------------------------
# 2. APP CONFIGURATION & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Fossil AI | Classifier", page_icon="🦕", initial_sidebar_state="expanded")

# NEW: Injected custom CSS function
def inject_custom_css():
    """Injects custom CSS for a sophisticated dark theme."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #0066ff;
        --secondary-color: #6366f1;
        --accent-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --background-primary: #0f0f23;
        --background-secondary: #1a1b3a;
        --background-tertiary: #2d2e5f;
        --text-primary: #ffffff;
        --text-secondary: #c7d2fe;
        --text-muted: #9ca3af;
        --border-color: #374151;
        --shadow-soft: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-medium: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        --shadow-large: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Global font and base styling */
    html, body, [class*="st-"] {
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        letter-spacing: -0.01em;
    }

    /* Dark theme background */
    .main {
        background: linear-gradient(135deg, var(--background-primary) 0%, var(--background-secondary) 100%);
        color: var(--text-primary);
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Enhanced sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--background-secondary) 0%, var(--background-tertiary) 100%);
        border-right: 1px solid var(--border-color);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    /* Sophisticated button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border: none;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.75rem 1.5rem;
        color: white;
        box-shadow: var(--shadow-soft);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
        box-shadow: var(--shadow-medium);
        transform: translateY(-2px);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-soft);
    }
    
    /* Primary button variant */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-color) 0%, #059669 100%);
        font-size: 1rem;
        padding: 1rem 2rem;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #059669 0%, var(--accent-color) 100%);
    }

    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--background-tertiary);
        border-radius: 16px;
        padding: 6px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-soft);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: transparent;
        border-radius: 12px;
        font-weight: 500;
        font-size: 0.9rem;
        color: var(--text-secondary);
        transition: all 0.3s ease;
        border: none;
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        font-weight: 600;
        box-shadow: var(--shadow-soft);
    }

    /* Typography enhancements */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'SF Pro Display', sans-serif;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    h1 { font-size: 2.5rem; margin-bottom: 1rem; }
    h2 { font-size: 2rem; margin-bottom: 0.875rem; }
    h3 { font-size: 1.5rem; margin-bottom: 0.75rem; }
    
    /* Enhanced header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 3rem;
        box-shadow: var(--shadow-large);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }

    /* Sophisticated card designs */
    .metric-card {
        background: linear-gradient(135deg, var(--background-tertiary) 0%, var(--background-secondary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        color: var(--text-primary);
        box-shadow: var(--shadow-medium);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-large);
        border-color: var(--primary-color);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        border-radius: 16px 16px 0 0;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        box-shadow: var(--shadow-large);
    }
    
    .success-card {
        background: linear-gradient(135deg, var(--accent-color) 0%, #059669 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        color: white;
        box-shadow: var(--shadow-medium);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .info-card {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        color: white;
        box-shadow: var(--shadow-medium);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, var(--warning-color) 0%, #d97706 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        color: white;
        box-shadow: var(--shadow-medium);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .dark-card {
        background: linear-gradient(135deg, var(--background-tertiary) 0%, var(--background-secondary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        color: var(--text-primary);
        box-shadow: var(--shadow-soft);
    }
    
    /* Enhanced metrics display */
    .stMetric {
        background: linear-gradient(135deg, var(--background-tertiary) 0%, var(--background-secondary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: var(--shadow-soft);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-medium);
    }
    
    .stMetric [data-testid="metric-container"] > div {
        color: var(--text-primary);
    }
    
    /* Enhanced form controls */
    .stSelectbox > div, .stMultiSelect > div {
        background: var(--background-tertiary);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    .stCheckbox > label {
        color: var(--text-primary);
        font-weight: 500;
    }
    
    .stRadio > label {
        color: var(--text-primary);
        font-weight: 500;
    }
    
    /* File uploader enhancement */
    .stFileUploader > div {
        background: var(--background-tertiary);
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-color);
        background: var(--background-secondary);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        border-radius: 8px;
    }
    
    /* Code blocks */
    .stCode {
        background: var(--background-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--background-tertiary);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        color: var(--text-primary);
        font-weight: 500;
    }
    
    /* Enhanced alerts */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: var(--shadow-soft);
    }
    
    .stAlert[data-baseweb="notification"] {
        background: linear-gradient(135deg, var(--background-tertiary) 0%, var(--background-secondary) 100%);
    }
    
    /* Plotly chart containers */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: var(--shadow-medium);
        overflow: hidden;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-secondary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    
    /* Loading animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Icon enhancements */
    .icon {
        display: inline-block;
        margin-right: 0.5rem;
        font-size: 1.2em;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .metric-card, .success-card, .info-card, .warning-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Call the function to apply the styles
inject_custom_css()
# ---------------------------------------------------------------------------
# 3. MODEL & DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI model... This may take a moment.")
def load_new_model_and_classes():
    try:
        # Works in normal .py scripts
        PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    except NameError:
        # Fallback for Jupyter notebooks
        PROJECT_ROOT = pathlib.Path.cwd().parent

    MODEL_DIR=PROJECT_ROOT / "4_Dashboard_App" / "fossil_classifier_final"

    MODEL_PATH = MODEL_DIR / "model.keras"; CLASSES_PATH = MODEL_DIR / "class_names.json"
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        st.error(f"Fatal Error: Model or class files not found in '{MODEL_DIR}'."); return None, None
    model = tf.keras.models.load_model(MODEL_PATH);
    with open(CLASSES_PATH) as f: class_names = json.load(f)
    return model, class_names

model, class_names = load_new_model_and_classes()
if model is None: st.stop()

# ---------------------------------------------------------------------------
# 4. HELPER FUNCTIONS & STATE MANAGEMENT
# ---------------------------------------------------------------------------
def initialize_session_state():
    defaults = {"raw_img": None, "predictions": None, "processed_img_to_display": None, "show_results": False, "trigger_prediction": False, "image_to_predict": None, "current_method": None, "last_method": None}
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

def reset_app_state():
    keys_to_delete = list(st.session_state.keys())
    for key in keys_to_delete:
        if key != 'page_config':
            del st.session_state[key]
    gc.collect()

def trigger_prediction_callback(image_to_process: np.ndarray):
    st.session_state.image_to_predict = image_to_process
    st.session_state.trigger_prediction = True

# NEW: A function dedicated to generating the preview image
def generate_preview(pil_image, method, threshold_adj):
    """Generates the final processed image for preview based on the current settings."""
    raw_img_gray = cv2.cvtColor(np.array(st.session_state.raw_img), cv2.COLOR_RGB2GRAY)
    is_white_background_case = np.mean(raw_img_gray) > 190
    
    img_to_process_gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    
    processed_np = None
    if method in ["Auto-Process", "Manual Crop"]:
        processed_np, success = segment_image(
            img_to_process_gray,
            assume_bright_fossil=not is_white_background_case,
            invert_output=is_white_background_case,
            threshold_adjustment=threshold_adj
        )
        if not success: return None
    else: # "Original Image"
        processed_np = img_to_process_gray
        
    if processed_np is not None:
        final_rgb_np = cv2.cvtColor(processed_np, cv2.COLOR_GRAY2RGB)
        img_tensor = tf.convert_to_tensor(final_rgb_np, dtype=tf.float32)
        cropped_tensor = crop_to_bbox(img_tensor)
        return cropped_tensor.numpy().astype("uint8")
    return None

# ---------------------------------------------------------------------------
# 5. SIDEBAR LAYOUT
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;">
            <h2 style="margin: 0; color: #ffffff; font-size: 1.5rem;">🦕 Fossil AI Classifier</h2>
            <p style="margin: 0.5rem 0 0 0; color: #c7d2fe; font-size: 0.9rem;">Deep Learning Classification System</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced reset button
    if st.button("🔄 Start Over with New Image", type="primary", use_container_width=True):
        reset_app_state()
        st.rerun()
    
    st.markdown("""
        <div style="height: 1px; background: linear-gradient(90deg, transparent, #374151, transparent); margin: 1rem 0;"></div>
    """, unsafe_allow_html=True)
    
    # Enhanced model details section
    st.markdown("""
        <div class="glass-card" style="padding: 0.8rem 1rem; margin-bottom: 0.5rem; text-align: center;">
            <h4 style="margin: 0; color: #ffffff; display: flex; align-items: center; justify-content: center;">
                <span class="icon">🧠</span>Model Architecture
            </h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Add spacing
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-card">
            <h5 style="margin: 0 0 0.75rem 0; color: white; font-weight: 600;">
                <span class="icon">⚡</span>Architecture Details
            </h5>
            <ul style="margin: 0; padding-left: 1.2rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; line-height: 1.4;">
                <li><strong>Ensemble Model:</strong> ConvNeXt-Large + EfficientNet-V2</li>
                <li><strong>Preprocessing:</strong> Adaptive segmentation & padding</li>
                <li><strong>Classes:</strong> 12 types of larger benthic foraminifera</li>
                <li><strong>Input Size:</strong> 384×384 pixels</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="height: 1px; background: linear-gradient(90deg, transparent, #374151, transparent); margin: 1rem 0;"></div>
    """, unsafe_allow_html=True)
    
    # Enhanced author and time info
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="margin-bottom: 1rem;">
                <p style="margin: 0; color: #c7d2fe; font-size: 0.85rem;">Created by</p>
                <p style="margin: 0; color: #ffffff; font-weight: 600; font-size: 0.9rem;">Dr. Abdelghafour Halimi</p>
            </div>
            <div class="dark-card" style="padding: 0.75rem; margin: 0;">
                <p style="margin: 0; color: #10b981; font-size: 0.8rem; font-weight: 500;">
                    🕒 Current Time (KAUST)<br>
                    <span style="color: #ffffff;">{}</span>
                </p>
            </div>
        </div>
    """.format(pd.Timestamp.now(tz="Asia/Riyadh").strftime("%I:%M %p - %B %d, %Y")), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 6. MAIN APPLICATION FLOW
# ---------------------------------------------------------------------------
initialize_session_state()

# Enhanced header with sophisticated design
st.markdown("""
    <div class="main-header">
        <div style="position: relative; z-index: 2;">
            <h1>🦕 AI-Powered Fossil Classification System</h1>
            <p style="font-size: 1.3rem; margin-bottom: 0.5rem;">Deep Learning Ensemble for Paleontological Identification</p>
            <p style="font-size: 1rem; opacity: 0.85;">
                Advanced ConvNeXt + EfficientNet architecture for accurate fossil species classification
            </p>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 12px; backdrop-filter: blur(10px);">
                <p style="margin: 0; font-size: 0.95rem;">
                    <span style="color: #00D4FF;">🖥️ Technology:</span> <strong style="color: #FFFFFF;">TensorFlow + Deep Learning</strong> |
                    <span style="color: #00FF88;">🔬 Architecture:</span> <strong style="color: #FFFFFF;">Ensemble AI Model </strong> |
                    <br>
                    <span style="color: #FFE66D;">🧠 Classes:</span> <strong style="color: #FFFFFF;">12 Benthic Foraminifera Species</strong>
                </p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="color: #c7d2fe; font-size: 1.1rem; margin: 0;">
            Upload a fossil image, adjust segmentation parameters, and get AI-powered species identification
        </p>
    </div>
""", unsafe_allow_html=True)

if not st.session_state.raw_img:
    # Enhanced file uploader section
    st.markdown("""
        <div class="glass-card" style="padding: 2rem; margin-bottom: 2rem;">
            <h3 style="margin: 0 0 1rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center; text-align: center;">
                <span class="icon">🖼️</span>Fossil Image Upload Center
            </h3>
            <p style="color: #c7d2fe; margin-bottom: 1.5rem; text-align: center;">
                Select a high-resolution fossil image for AI-powered classification
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image file (JPG, PNG, TIFF)", type=["jpg", "jpeg", "png", "tiff"], label_visibility="collapsed")
    if uploaded_file: 
        st.session_state.raw_img = Image.open(uploaded_file).convert('RGB')
        st.rerun()

if st.session_state.raw_img:
    st.markdown("""
        <div style="height: 1px; background: linear-gradient(90deg, transparent, #374151, transparent); margin: 2rem 0;"></div>
    """, unsafe_allow_html=True)
    
    # Enhanced step header
    st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h2 style="margin: 0 0 0.5rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center; text-align: center;">
                <span class="icon">⚙️</span>Step 1: Image Preparation & Processing
            </h2>
            <p style="color: #c7d2fe; font-size: 1.1rem; margin: 0; text-align: center;">
                Select processing method and fine-tune segmentation parameters for optimal results
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced processing method selection
    st.markdown("""
        <div class="glass-card" style="padding: 1.5rem; margin-bottom: 1.5rem;">
            <h4 style="margin: 0 0 1rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center;">
                <span class="icon">🔧</span>Processing Method Selection
            </h4>
        </div>
    """, unsafe_allow_html=True)
    
    method = st.radio(
        "Choose processing method:", 
        ["Auto-Process", "Manual Crop"], 
        horizontal=True, 
        key="method_selector",
        help="Auto-Process: Automatic fossil detection | Manual Crop: Define region manually"
    )
    
    if st.session_state.get("last_method") != method:
        if st.session_state.get("last_method") is not None: 
            st.session_state.show_results = False
        st.session_state.last_method = method
    
    # --- UI elements for processing ---
    image_to_process_pil = None
    if method == "Auto-Process":
        image_to_process_pil = st.session_state.raw_img

    elif method == "Manual Crop":
        st.markdown("""
            <div class="info-card" style="margin: 1rem 0;">
                <p style="margin: 0; color: white; font-weight: 500;">
                    🖱️ <strong>Manual Selection:</strong> Draw a bounding box around the fossil region for precise cropping
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        selection = image_selector(image=st.session_state.raw_img, selection_type="box", width=None, height=400)
        if selection and selection["selection"]["box"]:
            box = selection["selection"]["box"][0]
            x_min, x_max, y_min, y_max = box["x"][0], box["x"][1], box["y"][0], box["y"][1]
            image_to_process_pil = st.session_state.raw_img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

    if image_to_process_pil:
        st.markdown("""
            <div style="height: 1px; background: linear-gradient(90deg, transparent, #374151, transparent); margin: 2rem 0;"></div>
        """, unsafe_allow_html=True)
        
        # Enhanced step 2 header
        st.markdown("""
            <div style="margin-bottom: 2rem;">
                <h2 style="margin: 0 0 0.5rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center; text-align: center;">
                    <span class="icon">🎯</span>Step 2: Segmentation Adjustment
                </h2>
                <p style="color: #c7d2fe; font-size: 1.1rem; margin: 0; text-align: center;">
                    Fine-tune the segmentation threshold for optimal fossil boundary detection
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Enhanced segmentation controls
        st.markdown("""
            <div class="glass-card" style="padding: 1.5rem; margin-bottom: 1.5rem;">
                <h4 style="margin: 0 0 1rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center;">
                    <span class="icon">⚙️</span>Segmentation Sensitivity Control
                </h4>
            </div>
        """, unsafe_allow_html=True)
        
        # --- Interactive Segmentation Slider ---
        threshold_adj = st.slider(
            "Segmentation Sensitivity",
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            help="Fine-tune the segmentation threshold. Move left to include more (less severe), move right to exclude more (more severe)."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h5 style="margin: 0 0 0.1rem 0; color: #ffffff; text-align: center;">
                        📥 Input Image
                    </h5>
                </div>
            """, unsafe_allow_html=True)
            st.image(image_to_process_pil, caption=f"Input for {method}", use_container_width=True)
        
        # --- Live Preview ---
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h5 style="margin: 0 0 0.1rem 0; color: #ffffff; text-align: center;">
                        🔍 Processed Preview
                    </h5>
                </div>
            """, unsafe_allow_html=True)
            
            st.session_state.preview_image = generate_preview(image_to_process_pil, method, threshold_adj)
            if st.session_state.preview_image is not None:
                st.image(st.session_state.preview_image, caption="PREVIEW of Image to be Classified", use_container_width=True)
            else:
                st.markdown("""
                    <div class="warning-card">
                        <p style="margin: 0; color: white; font-weight: 500;">
                            ⚠️ <strong>Segmentation Failed:</strong> Current settings produced no valid result. Try adjusting the slider.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style="height: 1px; background: linear-gradient(90deg, transparent, #374151, transparent); margin: 2rem 0;"></div>
        """, unsafe_allow_html=True)
        
        # Enhanced classification button
        st.markdown("""
            <div style="text-align: center; margin: 2rem 0;">
                <h3 style="color: #ffffff; margin-bottom: 1rem;">
                    <span class="icon">🚀</span>Ready for AI Classification
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Classify This Fossil", use_container_width=True, type="primary"):
             if st.session_state.preview_image is not None:
                 trigger_prediction_callback(st.session_state.preview_image)
             else:
                 st.markdown("""
                     <div class="warning-card">
                         <p style="margin: 0; color: white; font-weight: 500;">
                             ❌ <strong>Cannot Classify:</strong> Segmentation failed. Please adjust the slider settings.
                         </p>
                     </div>
                 """, unsafe_allow_html=True)

    # --- Prediction Execution Logic ---
    if st.session_state.trigger_prediction:
        with st.spinner("🔬 Running AI model analysis..."):
            image_to_predict_np = st.session_state.image_to_predict
            image_batch = tf.expand_dims(tf.convert_to_tensor(image_to_predict_np, dtype=tf.float32), axis=0)
            predictions = model.predict(image_batch)
            st.session_state.predictions = predictions[0]
            st.session_state.processed_img_to_display = image_to_predict_np
            st.session_state.show_results = True
            st.session_state.trigger_prediction = False
            gc.collect()

    # --- Results Display ---
    if st.session_state.show_results and st.session_state.predictions is not None:
        st.markdown("""
            <div style="height: 2px; background: linear-gradient(90deg, transparent, #10b981, transparent); margin: 3rem 0;"></div>
        """, unsafe_allow_html=True)
        
        # Enhanced results header
        st.markdown("""
            <div style="margin-bottom: 2rem;">
                <h2 style="margin: 0 0 0.5rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center; text-align: center;">
                    <span class="icon">🧠</span>Step 3: AI Classification Results
                </h2>
                <p style="color: #c7d2fe; font-size: 1.1rem; margin: 0; text-align: center;">
                    Deep learning analysis complete - explore detailed predictions and confidence metrics
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        probs = st.session_state.predictions; sorted_indices = np.argsort(probs)[::-1]; top_index = sorted_indices[0]; top_class = class_names[top_index]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            # Enhanced image display
            st.markdown("""
                <div class="metric-card">
                    <h5 style="margin: 0 0 0.2rem 0; color: #ffffff; text-align: center; white-space: nowrap;">
                        🔬 Processed Input
                    </h5>
                </div>
            """, unsafe_allow_html=True)
            st.image(st.session_state.processed_img_to_display, caption="Image Fed to AI Model", use_container_width=True)
            
        with col2:
            # Enhanced top predictions table
            st.markdown("""
                <div class="metric-card">
                    <h4 style="margin: 0 0 0.2rem 0; color: #ffffff; text-align: center; white-space: nowrap;">
                        🏅 Top 5 Classification Results
                    </h4>
                </div>
            """, unsafe_allow_html=True)
            
            st.dataframe({
                "Rank": [f"#{i+1}" for i in range(5)], 
                "Fossil Species": [class_names[i] for i in sorted_indices[:5]], 
                "Confidence": [f"{probs[i]:.2%}" for i in sorted_indices[:5]]
            }, use_container_width=True, hide_index=True)
        
        # Enhanced metrics - side by side layout with original colors (full width)
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.markdown("""
                <div class="success-card">
                    <h5 style="margin: 0 0 0.75rem 0; color: white; font-weight: 600; text-align: center;">
                        🏆 Top Prediction
                    </h5>
                    <p style="margin: 0; font-size: 1.1rem; font-weight: 700; text-align: center;">{}</p>
                </div>
            """.format(top_class), unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown("""
                <div class="info-card">
                    <h5 style="margin: 0 0 0.75rem 0; color: white; font-weight: 600; text-align: center;">
                        📊 Confidence Level
                    </h5>
                    <p style="margin: 0; font-size: 1.1rem; font-weight: 700; text-align: center;">{:.2%}</p>
                </div>
            """.format(probs[top_index]), unsafe_allow_html=True)
        
        # Enhanced comprehensive analysis section
        st.markdown("""
            <div style="margin: 2rem 0;">
                <h3 style="margin: 0 0 0.5rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center; text-align: center;">
                    <span class="icon">🎯</span>Comprehensive Probability Analysis
                </h3>
                <p style="color: #c7d2fe; font-size: 1rem; margin: 0; text-align: center;">
                    Detailed confidence distribution and statistical insights
                </p>
            </div>
        """, unsafe_allow_html=True)
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 **Full Distribution**", "🔥 **Top Predictions**", "📈 **Confidence Metrics**"])
        with viz_tab1:
            percents = probs * 100; sorted_idx_plot = np.argsort(percents)
            colors = ['#10B981' if p >= 10 else '#F59E0B' if p >= 5 else '#3B82F6' if p >= 1 else '#6B7280' for p in percents[sorted_idx_plot]]
            fig = px.bar(x=percents[sorted_idx_plot], y=[class_names[i] for i in sorted_idx_plot], orientation='h', labels={"x": "Confidence (%)", "y": "Fossil Species"}, text=[f'{p:.2f}%' if p >= 0.1 else '' for p in percents[sorted_idx_plot]])
            fig.update_layout(title={'text': f"🔬 Complete Probability Distribution Across All {len(class_names)} Fossil Species", 'x': 0.5, 'xanchor': 'center'}, height=min(40 * len(class_names), 2800), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1E293B', xaxis=dict(range=[0, max(100, percents.max() + 5)], showgrid=True, gridcolor='#334155'), yaxis=dict(showgrid=False), bargap=0.3)
            fig.update_traces(marker_color=colors, textposition='outside', textfont_color='#E2E8F0', hovertemplate='<b>%{y}</b><br>Confidence: %{x:.3f}%<extra></extra>', marker_line=dict(width=0.5, color='rgba(255,255,255,0.1)'))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("<div style='background-color: #1E293B; padding: 1rem; border-radius: 8px; margin-top: 1rem;'><strong>🎨 Confidence Color Scale:</strong><br><span style='color: #10B981;'>● High (≥10%)</span> | <span style='color: #F59E0B;'>● Medium (5-10%)</span> | <span style='color: #3B82F6;'>● Low (1-5%)</span> | <span style='color: #6B7280;'>● Very Low (<1%)</span></div>", unsafe_allow_html=True)
        with viz_tab2:
            st.markdown("##### 🏆 Top 10 Most Likely Species"); top_10 = sorted_indices[:10]
            top_colors = ['#FFD700', '#C0C0C0', '#CD7F32'] + [f'rgba(59, 130, 246, {0.9 - (i * 0.08)})' for i in range(3, 10)]
            top_fig = px.bar(x=[probs[i]*100 for i in top_10], y=[class_names[i] for i in top_10], orientation='h', labels={"x": "Confidence (%)", "y": "Species"}, text=[f"{probs[i]*100:.2f}%" for i in top_10])
            # top_fig.update_layout(title={'text': "🏆 Top 10 Most Probable Classifications", 'x': 0.5, 'xanchor': 'center'}, height=600, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1E293B', yaxis=dict(autorange="reversed"))
            top_fig.update_layout(
                title={
                    'text': "🏆 Top 10 Most Probable Fossil Classifications",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                height=600,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#1E293B',
                xaxis=dict(
                    range=[0, 100],
                    showgrid=True, 
                    gridcolor='#334155',
                    title=dict(text="Confidence (%)", font=dict(size=14, color='#E2E8F0'))
                ),
                yaxis=dict(
                    showgrid=False, 
                    autorange="reversed",  # This ensures highest confidence is at top
                    title=dict(text="Fossil Species", font=dict(size=14, color='#E2E8F0'))
                ),
                bargap=0.2
            )
            top_fig.update_traces(marker_color=top_colors, textposition='outside', textfont_color='#FFFFFF', hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>')
            st.plotly_chart(top_fig, use_container_width=True)
            col_a, col_b, col_c = st.columns(3); confidence_gap = (probs[top_10[0]] - probs[top_10[1]]) * 100
            col_a.metric(label="🥇 Winner", value=class_names[top_10[0]], delta=f"{probs[top_10[0]]*100:.2f}%")
            col_b.metric(label="🥈 Runner-up", value=class_names[top_10[1]], delta=f"{probs[top_10[1]]*100:.2f}%")
            col_c.metric(label="📊 Confidence Gap", value=f"{confidence_gap:.2f}%", delta=f"{'High' if confidence_gap > 10 else 'Medium' if confidence_gap > 5 else 'Low'} certainty")
        with viz_tab3:
            st.markdown("##### 📊 Prediction Statistics & Analysis")
            entropy = -np.sum(probs * np.log(probs + 1e-10)); top_3_sum = np.sum(probs[sorted_indices[:3]]) * 100
            predictions_above_1, predictions_above_5 = np.sum(probs > 0.01), np.sum(probs > 0.05)
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            met_col1.metric("🎲 Entropy", f"{entropy:.2f}", "Lower = More Certain")
            met_col2.metric("🏆 Top 3 Combined", f"{top_3_sum:.1f}%", "Cumulative Confidence")
            met_col3.metric("📈 Species > 1%", f"{predictions_above_1}", "Viable Candidates")
            met_col4.metric("🔥 Species > 5%", f"{predictions_above_5}", "Strong Candidates")
            st.markdown("##### 📈 Confidence Distribution Analysis")
            hist_fig = px.histogram(x=probs*100, nbins=50, labels={"x": "Confidence (%)", "y": "Number of Species"}, color_discrete_sequence=['#3B82F6'])
            # hist_fig.update_layout(title={'text': "Distribution of Prediction Confidences", 'x': 0.5}, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1E293B', height=450)
            hist_fig.update_layout(
                title={
                    'text': "📊 Distribution of Prediction Confidences",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='#1E293B',
                height=450,
                xaxis=dict(
                    range=[0, 100],
                    showgrid=True, 
                    gridcolor='#334155',
                    title=dict(text="Confidence (%)", font=dict(size=14, color='#E2E8F0'))
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridcolor='#334155',
                    title=dict(text="Number of Species", font=dict(size=14, color='#E2E8F0'))
                )
            )
            st.plotly_chart(hist_fig, use_container_width=True)
                        # Add interpretation for the histogram
            st.markdown("""
            <div style='background-color: #1E293B; padding: 1rem; border-radius: 8px; margin-top: 1rem; border-left: 4px solid #3B82F6;'>
                <strong>📈 Chart Interpretation:</strong><br>
                This histogram shows how prediction confidence is distributed across all 12 fossil species. 
                A tall peak near 0% indicates many species have very low confidence (which is expected), 
                while bars on the right show how many species have higher confidence levels. 
                <strong>The shape tells you about model certainty:</strong> a sharp peak at low values with few high-confidence species suggests focused predictions.
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretation guide
            st.markdown("""
            <div style='background-color: #1E293B; padding: 1.5rem; border-radius: 12px; border-left: 4px solid #3B82F6;'>
                <h4 style='color: #FFFFFF; margin-top: 0;'>🧠 How to Interpret These Results:</h4>
                <ul style='color: #E2E8F0; line-height: 1.6;'>
                    <li><strong>Entropy:</strong> Values closer to 0 indicate higher model confidence. Values above 4 suggest uncertainty.</li>
                    <li><strong>Top 3 Combined:</strong> Higher percentages indicate the model is confident in a few species.</li>
                    <li><strong>Viable Candidates:</strong> Species with >1% confidence are worth considering for identification.</li>
                    <li><strong>Strong Candidates:</strong> Species with >5% confidence represent the most likely classifications.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
