import streamlit as st
import numpy as np
import os, tempfile, gc, math, sys
from PIL import Image
from skimage.color import rgb2gray
from skimage import io
from skimage.transform import resize
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import cv2 # Make sure this import is at the top of your file
from skimage import filters, morphology, measure
import pathlib

# Configure page
st.set_page_config(
    page_title="🦕 3D Fossil Slice Matcher",
    page_icon="🦕",
    # layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling for sophisticated UI
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
    
    /* Species and result cards */
    .species-card {
        background: linear-gradient(135deg, var(--background-tertiary) 0%, var(--background-secondary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        color: var(--text-primary);
        box-shadow: var(--shadow-soft);
        transition: all 0.3s ease;
    }
    
    .species-card:hover {
        border-color: var(--primary-color);
        box-shadow: var(--shadow-medium);
    }
    
    .match-result {
        background: linear-gradient(135deg, var(--accent-color) 0%, #059669 100%);
        border-left: 4px solid #10b981;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 0.75rem 0;
        color: white;
        box-shadow: var(--shadow-medium);
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
    
    /* Glassmorphism effect for special elements */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        box-shadow: var(--shadow-large);
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

# Import all the necessary modules
try:
    import nibabel as nib
except ImportError:
    st.error("nibabel package is required. Install with: pip install nibabel")
    st.stop()

try:
    from streamlit_extras.image_selector import image_selector, show_selection
except ImportError:
    st.warning("streamlit_extras not found. Image cropping will be disabled.")
    image_selector = None
    show_selection = None

# --- FIX: Added robust device configuration with fallback for CUDA errors ---
try:
    if torch.cuda.is_available():
        # Try CUDA:1 first
        try:
            device = torch.device("cuda:1")
            torch.tensor([1.0]).to(device)
            print(f"✅ Successfully initialized CUDA device: {torch.cuda.get_device_name(device)}")
        except Exception:
            # Fallback to CUDA:0
            device = torch.device("cuda:0")
            torch.tensor([1.0]).to(device)
            print(f"✅ Successfully initialized CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("ℹ️ CUDA not available, using CPU.")
except RuntimeError as e:
    if "no kernel image is available" in str(e):
        print("⚠️ CUDA compatibility error. Falling back to CPU. Your PyTorch version may not match your GPU's architecture.")
        print("➡️ For GPU acceleration, please reinstall PyTorch using the command from https://pytorch.org/")
        device = torch.device("cpu")
    else:
        # Re-raise other runtime errors
        raise e
except Exception as e:
    print(f"An unexpected error occurred during device configuration: {e}")
    device = torch.device("cpu")

# Enhanced header with sophisticated design
st.markdown("""
    <div class="main-header">
        <div style="position: relative; z-index: 2;">
            <h1>🦕 AI-Powered 3D Fossil Identification System</h1>
            <p style="font-size: 1.3rem; margin-bottom: 0.5rem;">Advanced Deep Learning for Paleontological Analysis</p>
            <p style="font-size: 1rem; opacity: 0.85;">
                Match 2D fossil slices against comprehensive 3D model database using SSIM + NCC similarity
            </p>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 12px; backdrop-filter: blur(10px);">
                <p style="margin: 0; font-size: 0.95rem;">
                    <span style="color: #10b981;">🖥️ Device:</span> <strong>{}</strong> |
                    <span style="color: #10b981;">🔬 Technology:</span> PyTorch + Computer Vision |
                    <span style="color: #10b981;">🧠 AI Engine:</span> Multi-metric Similarity Analysis
                </p>
            </div>
        </div>
    </div>
""".format(device), unsafe_allow_html=True)

# Preset configurations
ROTATION_ANGLES = [20, 40, 60, 80, 100, 120, 140, 160]

# All utility functions from original code
def get_species_from_filename(filename):
    """Extract species name from filename using comprehensive pattern matching"""
    # Comprehensive species patterns based on the 27 species in the analysis
    species_patterns = [
        "Alveolina", "Amphistegina", "Arumella", "Ataxophragmium", "Baculogypsina",
        "Chrysalidina", "Cincoriola", "Coscinoidea", "Coskinolina", "Daviesina",
        "Dictyoconoides", "Dictyoconus", "Dukhania", "Elphidiella", "Fallotia",
        "Fissolephidium", "Glyphostomalloides", "Heterostegina", "Lockhartia",
        "Minoxia", "Miscellanea", "Nummulites", "Omphalocyclus", "Orbitoides",
        "Peneroplis", "Rhapydionina", "Rotalia"
    ]
    
    # Check each pattern in the filename (case-insensitive)
    filename_lower = filename.lower()
    for pattern in species_patterns:
        if pattern.lower() in filename_lower:
            return pattern
    
    # Special cases for tricky names
    if "dictyoconides" in filename_lower:
        return "Dictyoconoides"
    if "omphaalocyclus" in filename_lower:  # Handle typo in filename
        return "Omphalocyclus"
    
    return "Unknown_Species"

def dynamic_downsample(volume, max_dim=128):
    fz = fy = fx = 10
    return volume[::fz, ::fy, ::fx], (fz, fy, fx)

def ensure_non_degenerate_range(vmin, vmax, padding=1.0):
    if vmin >= vmax:
        mid = (vmin + vmax) / 2.0
        return (mid - padding, mid + padding)
    return (vmin, vmax)

def resample_isotropic(vol, hdr):
    """Resample volume to isotropic voxels"""
    zooms = np.array(hdr.get_zooms()[:3])
    min_zoom = min(zooms)
    zoom_factors = zooms / min_zoom
    if zoom_factors.ptp() > 1e-3:
        return zoom(vol, zoom_factors, order=1)
    return vol

def smart_threshold(vol):
    """Smart thresholding"""
    from skimage.filters import threshold_otsu
    mx = vol.max()
    nz = vol[vol > vol.min()]
    if nz.size < 10:
        return 0.4 * mx
    try:
        thr = threshold_otsu(nz)
    except ValueError:
        thr = 0.4 * mx
    return 0.4 * mx if thr > 0.8 * mx else thr

def build_fossil_mask(vol, thr):
    """Build fossil mask"""
    from scipy.ndimage import label
    lbl, n = label(vol > thr)
    if n == 0: 
        return np.ones_like(vol, bool), (0,) * 6
    sizes = np.bincount(lbl.ravel())[1:]
    mask = lbl == (np.argmax(sizes) + 1)
    loc = np.where(mask)
    x0, x1 = loc[0].min(), loc[0].max()
    y0, y1 = loc[1].min(), loc[1].max()
    z0, z1 = loc[2].min(), loc[2].max()
    BOUNDING_PAD = 2
    return mask, (max(0, x0-BOUNDING_PAD), min(vol.shape[0]-1, x1+BOUNDING_PAD),
                  max(0, y0-BOUNDING_PAD), min(vol.shape[1]-1, y1+BOUNDING_PAD),
                  max(0, z0-BOUNDING_PAD), min(vol.shape[2]-1, z1+BOUNDING_PAD))

def slice_has_fossil(mask2d, min_area_ratio=0.05):
    """Check if slice has enough fossil content - stricter filtering"""
    fossil_ratio = mask2d.sum() / mask2d.size
    # Also check for connected fossil regions (avoid scattered noise)
    if fossil_ratio < min_area_ratio:
        return False
    
    # Additional check: ensure we have a substantial connected component
    from scipy.ndimage import label
    labeled, num_features = label(mask2d)
    if num_features == 0:
        return False
    
    # Check that the largest component is substantial
    component_sizes = np.bincount(labeled.ravel())[1:]  # Exclude background
    largest_component_ratio = np.max(component_sizes) / mask2d.size
    
    return largest_component_ratio >= min_area_ratio * 0.7  # At least 70% of min_area should be in one component

def valid_idx(mask, axis, bbox=None):
    """Get valid slice indices that have fossil content - now with bounding box focus"""
    if bbox is not None:
        # Focus only on slices within the fossil bounding box
        x0, x1, y0, y1, z0, z1 = bbox
        if axis == 0:  # Z-axis slices
            valid_range = range(max(0, z0), min(mask.shape[0], z1 + 1))
            valid_indices = [i for i in valid_range if mask[i, :, :].any()]
        elif axis == 1:  # Y-axis slices  
            valid_range = range(max(0, y0), min(mask.shape[1], y1 + 1))
            valid_indices = [i for i in valid_range if mask[:, i, :].any()]
        else:  # X-axis slices
            valid_range = range(max(0, x0), min(mask.shape[2], x1 + 1))
            valid_indices = [i for i in valid_range if mask[:, :, i].any()]
        return np.array(valid_indices)
    else:
        # Fallback to original method
        if axis == 0: 
            return np.where(mask.any(axis=(1, 2)))[0]
        if axis == 1: 
            return np.where(mask.any(axis=(0, 2)))[0]
        return np.where(mask.any(axis=(0, 1)))[0]

def volume_minmax_percentile(vol):
    """Get volume min/max using percentiles"""
    p1, p99 = np.percentile(vol, [1, 99])
    v = np.clip(vol, p1, p99)
    return v.min(), v.max()

def normalize_to_01_torch(tensor):
    min_val, max_val = torch.min(tensor), torch.max(tensor)
    return tensor - min_val if (max_val - min_val) < 1e-8 else \
           (tensor - min_val) / (max_val - min_val + 1e-8)

def apply_fossil_segmentation(slice_np, threshold=0.5, show_preview=False, return_mask=False, preserve_holes=True):
    """
    Enhanced fossil segmentation that preserves internal structure and holes.
    
    IMPROVED: Now preserves internal chambers, pores, and holes instead of filling them.
    This maintains the diagnostic structural features that distinguish different fossil species.
    
    Args:
        slice_np: Input grayscale image
        threshold: Segmentation sensitivity 
        show_preview: Whether to show segmentation preview
        return_mask: Whether to return the binary mask
        preserve_holes: Whether to preserve internal holes/chambers (NEW)
    
    Returns:
        segmented: Segmented image with background removed
        mask: Binary mask (if return_mask=True)
    """
    try:
        from skimage import filters, morphology, measure, segmentation
        from scipy import ndimage
        
        original = slice_np.copy()
        
        # Enhanced thresholding approach for better fossil isolation
        # Method 1: Otsu thresholding (automatic)
        try:
            otsu_threshold = filters.threshold_otsu(slice_np)
            mask_otsu = slice_np > otsu_threshold
        except ValueError:
            mask_otsu = slice_np > np.mean(slice_np)
        
        # Method 2: User-controlled threshold (more aggressive)
        global_mean = np.mean(slice_np)
        global_std = np.std(slice_np)
        adaptive_threshold = global_mean + (threshold * 2.0) * global_std
        mask_adaptive = slice_np > adaptive_threshold
        
        # Method 3: Percentile-based thresholding (focus on top intensities)
        percentile_threshold = np.percentile(slice_np, 60 + threshold * 35)
        mask_percentile = slice_np > percentile_threshold
        
        # Method 4: Histogram-based approach (isolate fossil peaks)
        hist, bins = np.histogram(slice_np.ravel(), bins=50)
        peak_indices = np.where(hist > np.max(hist) * 0.1)[0]
        if len(peak_indices) > 1:
            fossil_threshold = bins[peak_indices[-1]]
            mask_histogram = slice_np > fossil_threshold
        else:
            mask_histogram = mask_otsu
        
        # Combine masks with weighted voting (more conservative approach)
        mask_votes = (mask_otsu.astype(int) + 
                     mask_adaptive.astype(int) + 
                     mask_percentile.astype(int) + 
                     mask_histogram.astype(int))
        
        # Require at least 3 out of 4 methods to agree for fossil pixels
        mask_combined = mask_votes >= 3
        
        # If too restrictive, fallback to 2/4 agreement
        if np.sum(mask_combined) < slice_np.size * 0.05:
            mask_combined = mask_votes >= 2
        
        # IMPROVED: Structure-preserving cleanup instead of aggressive hole filling
        if np.sum(mask_combined) > 0:
            # Remove very small noise objects but preserve larger internal structures
            mask_cleaned = morphology.remove_small_objects(mask_combined, min_size=50)  # Reduced from 200
        else:
            mask_cleaned = mask_combined
        
        # NEW: Conditional hole filling - preserve internal chambers and pores
        if preserve_holes:
            # Only fill very small holes (likely noise) but preserve larger internal structures
            # Use a much smaller area threshold to maintain fossil architecture
            try:
                # Fill only tiny holes (noise) but preserve chambers/pores
                mask_filled = morphology.remove_small_holes(mask_cleaned, area_threshold=20)  # Very small threshold
                
                # Additional check: if hole filling removed too much structure, revert
                original_holes = np.sum(~mask_cleaned & (mask_cleaned | morphology.binary_dilation(mask_cleaned, morphology.disk(2))))
                filled_holes = np.sum(mask_filled) - np.sum(mask_cleaned)
                
                # If we filled too many holes (more than 5% of image), it's probably removing important structure
                if filled_holes > slice_np.size * 0.05:
                    # st.write("  🔍 Preserving internal structure - skipping aggressive hole filling")
                    mask_filled = mask_cleaned
                
            except Exception:
                # Fallback to no hole filling if operation fails
                mask_filled = mask_cleaned
        else:
            # Original aggressive hole filling for comparison
            mask_filled = ndimage.binary_fill_holes(mask_cleaned)
        
        # Get connected components and analyze them
        labeled = measure.label(mask_filled)
        if labeled.max() > 0:
            # Analyze component properties to select the best fossil representation
            props = measure.regionprops(labeled, intensity_image=original)
            
            # Sort components by a combination of size and intensity
            component_scores = []
            for prop in props:
                # Score based on area, mean intensity, and shape characteristics
                area_score = prop.area
                intensity_score = prop.mean_intensity if hasattr(prop, 'mean_intensity') else np.mean(original[labeled == prop.label])
                
                # Prefer components that are not too elongated (more fossil-like)
                if prop.major_axis_length > 0:
                    aspect_ratio = prop.minor_axis_length / prop.major_axis_length
                    shape_score = min(aspect_ratio, 1.0)  # Prefer more compact shapes
                else:
                    shape_score = 1.0
                
                # Combined score favoring larger, brighter, more compact components
                combined_score = area_score * intensity_score * shape_score
                component_scores.append((prop.label, combined_score, area_score))
            
            # Select the best component(s)
            if component_scores:
                # Sort by combined score
                component_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Include the best component and any other substantial components
                final_mask = np.zeros_like(labeled, dtype=bool)
                best_score = component_scores[0][1]
                best_area = component_scores[0][2]
                
                for label, score, area in component_scores:
                    # Include components that are either very good or substantial in size
                    if score >= best_score * 0.3 and area >= best_area * 0.1:
                        final_mask |= (labeled == label)
                    
                    # Don't include too many small components
                    if np.sum(final_mask) > slice_np.size * 0.8:
                        break
            else:
                final_mask = mask_filled
        else:
            final_mask = mask_filled
        
        # IMPROVED: Gentler morphological operations to preserve fine structure
        # Use smaller structuring elements to avoid destroying internal features
        final_mask = morphology.binary_opening(final_mask, morphology.disk(1))  # Reduced from 2
        
        # Only apply closing if we're not preserving holes, or use very gentle closing
        if preserve_holes:
            # Very gentle closing to smooth boundaries without filling important holes
            final_mask = morphology.binary_closing(final_mask, morphology.disk(2))
        else:
            # Original aggressive closing
            final_mask = morphology.binary_closing(final_mask, morphology.disk(4))
        
        # Apply the mask to extract fossil content with preserved internal structure
        segmented = original * final_mask
        
        # Enhanced preview with structural analysis
        if show_preview:
            segmented_display = np.zeros_like(original)
            segmented_display[final_mask] = original[final_mask]
            
            # Create comparison view
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(original, caption="📸 Original Image", use_container_width=True)
            
            with col2:
                st.image(final_mask.astype(float), caption="🎯 Structure-Preserving Mask", use_container_width=True)
            
            with col3:
                st.image(segmented_display, caption="✨ Segmented with Internal Structure", use_container_width=True)
            
            # Enhanced statistics with structural analysis
            fossil_pixels = np.sum(final_mask)
            total_pixels = final_mask.size
            fossil_percentage = (fossil_pixels / total_pixels) * 100
            
            # Analyze internal structure preservation
            if preserve_holes:
                # Count holes/chambers (connected components in the inverse mask within the fossil region)
                inverse_mask = ~final_mask
                fossil_bbox = measure.regionprops(final_mask.astype(int))[0].bbox if np.any(final_mask) else (0, 0, final_mask.shape[0], final_mask.shape[1])
                roi_inverse = inverse_mask[fossil_bbox[0]:fossil_bbox[2], fossil_bbox[1]:fossil_bbox[3]]
                
                try:
                    internal_holes = measure.label(roi_inverse)
                    num_holes = internal_holes.max()
                    
                    if num_holes > 0:
                        hole_areas = [np.sum(internal_holes == i) for i in range(1, num_holes + 1)]
                        significant_holes = len([area for area in hole_areas if area > 10])  # Holes larger than 10 pixels
                    else:
                        significant_holes = 0
                        
                except Exception:
                    significant_holes = 0
                    num_holes = 0
                
                structure_info = f"🏛️ **Internal Structure Preserved:**\n" \
                               f"- Total internal features: {num_holes}\n" \
                               f"- Significant chambers/pores: {significant_holes}\n" \
                               f"- Hole preservation mode: {'ENABLED' if preserve_holes else 'DISABLED'}"
            else:
                structure_info = "🔄 **Aggressive hole filling:** Internal chambers filled"
            
            # Calculate intensity statistics
            fossil_intensity = np.mean(original[final_mask]) if fossil_pixels > 0 else 0
            background_intensity = np.mean(original[~final_mask]) if np.sum(~final_mask) > 0 else 0
            contrast_ratio = fossil_intensity / (background_intensity + 1e-8)
            
            st.success(f"🎯 **Enhanced Structure-Preserving Segmentation:**\n"
                      f"- Fossil pixels: {fossil_pixels:,} ({fossil_percentage:.1f}% of image)\n"
                      f"- Fossil intensity: {fossil_intensity:.3f}\n"
                      f"- Background intensity: {background_intensity:.3f}\n"
                      f"- Contrast ratio: {contrast_ratio:.2f}x\n"
                      f"{structure_info}")
        
        if return_mask:
            return segmented, final_mask
        return segmented
        
    except ImportError:
        # Fallback if scikit-image is not available
        st.warning("⚠️ Enhanced segmentation requires scikit-image. Using simple thresholding.")
        threshold_val = np.mean(slice_np) + threshold * np.std(slice_np)
        mask = slice_np > threshold_val
        
        # Even in fallback, try to preserve some structure
        if preserve_holes:
            # Don't fill holes in fallback mode
            pass
        else:
            # Fill holes only if requested
            try:
                mask = ndimage.binary_fill_holes(mask)
            except Exception:
                pass
        
        segmented = slice_np * mask
        if return_mask:
            return segmented, mask
        return segmented
    
    except Exception as e:
        st.error(f"Segmentation failed: {e}. Using original image.")
        if return_mask:
            return slice_np, np.ones_like(slice_np, dtype=bool)
        return slice_np

def new_segment_image(img_gray, assume_bright_fossil=True, invert_output=False, threshold_adjustment=0.0) -> tuple[np.ndarray | None, bool]:
    """
    This is the robust segmentation function from the previous app.
    It takes a numpy array, segments it based on the slider, inverts if needed,
    and returns the processed image and a success flag.
    """
    try:
        if img_gray is None or img_gray.size == 0:
            return None, False
        
        # Convert to 8-bit if it's a float image for OpenCV compatibility
        if img_gray.dtype != np.uint8:
            img_gray = (255 * (img_gray - np.min(img_gray)) / (np.max(img_gray) - np.min(img_gray) + 1e-8)).astype(np.uint8)

        blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
        
        # Adjust the Otsu threshold using the slider value
        otsu_thresh = filters.threshold_otsu(blurred)
        adjusted_thresh = otsu_thresh + threshold_adjustment
        
        binary = blurred > adjusted_thresh if assume_bright_fossil else blurred < adjusted_thresh
        
        # Use skimage morphology functions
        cleaned = morphology.remove_small_objects(binary, min_size=200)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=100)
        cleaned = morphology.binary_closing(cleaned, morphology.disk(2))

        # Use skimage measure function
        labeled = measure.label(cleaned)
        if labeled.max() == 0:
            return None, False

        main_fossil_label = max(measure.regionprops(labeled), key=lambda x: x.area).label
        mask = (labeled == main_fossil_label).astype(np.uint8) * 255
        
        # Use the original grayscale image for masking to preserve texture
        segmented_gray = cv2.bitwise_and(img_gray, img_gray, mask=mask)

        if invert_output:
            non_black_mask = segmented_gray > 0
            segmented_gray[non_black_mask] = 255 - segmented_gray[non_black_mask]

        return segmented_gray.astype(np.float32), True
    except Exception as e:
        st.error(f"Error during segmentation: {e}")
        return None, False


def segment_volume_slice(volume_slice, threshold=0.5, preserve_holes=True):
    """
    Apply enhanced segmentation to volume slices for fair fossil-to-fossil comparison.
    Uses the same aggressive approach as input slice segmentation.
    """
    try:
        from skimage import filters, morphology, measure
        from scipy import ndimage
        
        # Convert from torch tensor if needed
        if hasattr(volume_slice, 'cpu'):
            slice_np = volume_slice.cpu().numpy().squeeze()
            device = volume_slice.device
        else:
            slice_np = volume_slice.squeeze()
            device = None
        
        original = slice_np.copy()
        
        # Enhanced segmentation matching the input slice approach
        # Method 1: Otsu thresholding
        try:
            otsu_threshold = filters.threshold_otsu(slice_np)
            mask_otsu = slice_np > otsu_threshold
        except ValueError:
            mask_otsu = slice_np > np.mean(slice_np)
        
        # Method 2: Aggressive adaptive thresholding (same as input)
        global_mean = np.mean(slice_np)
        global_std = np.std(slice_np)
        adaptive_threshold = global_mean + (threshold * 2.0) * global_std  # Doubled sensitivity
        mask_adaptive = slice_np > adaptive_threshold
        
        # Method 3: Enhanced percentile-based thresholding
        percentile_threshold = np.percentile(slice_np, 60 + threshold * 35)  # 60-95% range
        mask_percentile = slice_np > percentile_threshold
        
        # Method 4: Histogram-based fossil detection
        hist, bins = np.histogram(slice_np.ravel(), bins=50)
        peak_indices = np.where(hist > np.max(hist) * 0.1)[0]
        if len(peak_indices) > 1:
            fossil_threshold = bins[peak_indices[-1]]
            mask_histogram = slice_np > fossil_threshold
        else:
            mask_histogram = mask_otsu
        
        # Combine masks with weighted voting (require 3/4 agreement)
        mask_votes = (mask_otsu.astype(int) + 
                     mask_adaptive.astype(int) + 
                     mask_percentile.astype(int) + 
                     mask_histogram.astype(int))
        
        mask_combined = mask_votes >= 3
        
        # Fallback to 2/4 agreement if too restrictive
        if np.sum(mask_combined) < slice_np.size * 0.05:
            mask_combined = mask_votes >= 2
        
        # Enhanced cleanup for volume slices
        if np.sum(mask_combined) > 0:
            # Remove noise but be more lenient for volume slices
            mask_cleaned = morphology.remove_small_objects(mask_combined, min_size=100)
        else:
            mask_cleaned = mask_combined
        
        # Conditional hole filling - preserve internal structure if requested
        if preserve_holes:
            # Only fill very small holes (likely noise) but preserve larger internal structures
            try:
                mask_filled = morphology.remove_small_holes(mask_cleaned, area_threshold=20)
            except Exception:
                mask_filled = mask_cleaned
        else:
            # Original aggressive hole filling
            mask_filled = ndimage.binary_fill_holes(mask_cleaned)
        
        # Handle multiple fossil components in volume slices
        labeled = measure.label(mask_filled)
        if labeled.max() > 0:
            component_sizes = np.bincount(labeled.flat)[1:]
            
            # Include multiple components if they're significant
            # Volume slices might have multiple fossil parts
            total_fossil_area = np.sum(mask_filled)
            min_component_size = max(50, total_fossil_area * 0.1)  # At least 10% of total
            
            final_mask = np.zeros_like(labeled, dtype=bool)
            for comp_idx in range(1, labeled.max() + 1):
                component_mask = (labeled == comp_idx)
                if np.sum(component_mask) >= min_component_size:
                    final_mask |= component_mask
            
            # If no significant components, use the largest one
            if not np.any(final_mask) and len(component_sizes) > 0:
                largest_idx = component_sizes.argmax() + 1
                final_mask = (labeled == largest_idx)
        else:
            final_mask = mask_filled
        
        # Enhanced morphological operations
        final_mask = morphology.binary_opening(final_mask, morphology.disk(2))
        final_mask = morphology.binary_closing(final_mask, morphology.disk(4))
        
        # Apply the mask for pure fossil content
        segmented = original * final_mask
        
        # Convert back to torch tensor if needed
        if device is not None:
            return torch.from_numpy(segmented[None, None]).float().to(device)
        else:
            return segmented
            
    except ImportError:
        # Enhanced fallback
        slice_np = volume_slice.cpu().numpy().squeeze() if hasattr(volume_slice, 'cpu') else volume_slice.squeeze()
        
        # More aggressive fallback thresholding
        global_mean = np.mean(slice_np)
        global_std = np.std(slice_np)
        threshold_val = global_mean + (threshold * 1.5) * global_std  # More aggressive
        mask = slice_np > threshold_val
        segmented = slice_np * mask
        
        if hasattr(volume_slice, 'cpu'):
            return torch.from_numpy(segmented[None, None]).float().to(volume_slice.device)
        else:
            return segmented
    
    except Exception as e:
        # Return original if segmentation fails
        return volume_slice
    
    except Exception:
        # Return original if anything fails
        return volume_slice

def detect_fossil_in_2d_image(image_np, padding=10):
    """Detect fossil content in 2D image and return bounding box for cropping"""
    from skimage.filters import threshold_otsu
    from scipy.ndimage import label
    
    # Method 1: Otsu thresholding
    try:
        threshold_otsu_val = threshold_otsu(image_np)
        binary_mask_otsu = image_np > threshold_otsu_val
    except ValueError:
        binary_mask_otsu = None
    
    # Method 2: Percentile-based thresholding (more robust)
    threshold_percentile = np.percentile(image_np, 75)  # Top 25% of intensities
    binary_mask_percentile = image_np > threshold_percentile
    
    # Method 3: Standard deviation based (finds areas with high contrast)
    mean_val = np.mean(image_np)
    std_val = np.std(image_np)
    threshold_std = mean_val + 0.5 * std_val
    binary_mask_std = image_np > threshold_std
    
    # Try each method and pick the one with the most reasonable result
    methods = [
        ("otsu", binary_mask_otsu),
        ("percentile", binary_mask_percentile), 
        ("std", binary_mask_std)
    ]
    
    best_bbox = None
    best_area_ratio = 0
    best_method = None
    
    for method_name, binary_mask in methods:
        if binary_mask is None:
            continue
            
        # Find connected components
        labeled, num_features = label(binary_mask)
        
        if num_features == 0:
            continue
        
        # Find the largest connected component
        component_sizes = np.bincount(labeled.ravel())[1:]  # Exclude background
        largest_component = np.argmax(component_sizes) + 1
        
        # Get bounding box of the largest component
        fossil_mask = (labeled == largest_component)
        rows, cols = np.where(fossil_mask)
        
        if len(rows) == 0 or len(cols) == 0:
            continue
        
        # Calculate bounding box with padding
        y_min = max(0, rows.min() - padding)
        y_max = min(image_np.shape[0] - 1, rows.max() + padding)
        x_min = max(0, cols.min() - padding)
        x_max = min(image_np.shape[1] - 1, cols.max() + padding)
        
        # Check if the detected region is reasonable
        detected_area = (x_max - x_min + 1) * (y_max - y_min + 1)
        total_area = image_np.shape[0] * image_np.shape[1]
        area_ratio = detected_area / total_area
        
        # Look for regions that are substantial but not the whole image
        if 0.05 <= area_ratio <= 0.85:  # Between 5% and 85% of image
            if area_ratio > best_area_ratio:
                best_bbox = (x_min, x_max, y_min, y_max)
                best_area_ratio = area_ratio
                best_method = method_name
    
    if best_bbox is not None:
        x_min, x_max, y_min, y_max = best_bbox
        return x_min, x_max, y_min, y_max, True
    else:
        # No good detection found, return original image bounds
        return 0, image_np.shape[1]-1, 0, image_np.shape[0]-1, False

# =================================================================
# Similarity metrics and slice extraction functions
# =================================================================
def create_gaussian_window(window_size=11, sigma=1.5, channels=1):
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= (window_size - 1) / 2.0
    gauss  = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    gauss2d = (gauss.unsqueeze(0) * gauss.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return gauss2d.expand(channels, 1, window_size, window_size)

def ssim_torch(img1, img2, window_size: int = 11, data_range: float = 1.0):
    """Structural SIMilarity for two single‑channel images"""
    try:
        if not torch.is_tensor(img1):
            img1 = torch.from_numpy(img1).float()
        if not torch.is_tensor(img2):
            img2 = torch.from_numpy(img2).float()

        if img1.dim() == 2: 
            img1 = img1[None, None]
        if img2.dim() == 2: 
            img2 = img2[None, None]
        img1, img2 = img1.to(img2.dtype), img2

        # Ensure images are on the same device
        img1 = img1.to(img2.device)

        # Normalize to [0, data_range] range
        img1 = (img1 - img1.min()) / (img1.max() - img1.min() + 1e-8) * data_range
        img2 = (img2 - img2.min()) / (img2.max() - img2.min() + 1e-8) * data_range

        window = create_gaussian_window(window_size, sigma=1.5).to(img1.device)

        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        pad = window_size // 2

        mu1 = F.conv2d(img1, weight=window, bias=None, stride=1, padding=pad, groups=1)
        mu2 = F.conv2d(img2, weight=window, bias=None, stride=1, padding=pad, groups=1)

        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, weight=window, bias=None, stride=1, padding=pad, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, weight=window, bias=None, stride=1, padding=pad, groups=1) - mu2_sq
        sigma12   = F.conv2d(img1*img2, weight=window, bias=None, stride=1, padding=pad, groups=1) - mu1_mu2

        num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
        den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / den
        
        # SSIM should be in range [0, 1], not [-1, 1]
        ssim_value = ssim_map.mean().clamp(0, 1).item()
        return ssim_value
        
    except Exception as e:
        # Return a reasonable default value if computation fails
        print(f"SSIM computation error: {e}")
        return 0.0

def ncc_torch(img1, img2):
    f1, f2 = img1.view(-1), img2.view(-1)
    mean1 = torch.mean(f1); mean2 = torch.mean(f2)
    num = torch.sum((f1-mean1)*(f2-mean2))
    denom = torch.sqrt(torch.sum((f1-mean1)**2)*torch.sum((f2-mean2)**2) + 1e-8)
    return (num/denom).item()

def combined_similarity_torch(ssim_val, ncc_val, w_ssim=0.5, w_ncc=0.5):
    ssim_norm = (ssim_val+1)/2; ncc_norm = (ncc_val+1)/2
    return w_ssim*ssim_norm + w_ncc*ncc_norm

# =================================================================
# NEW: Two-Stage Coarse-to-Fine Matching Pipeline
# =================================================================

def dice_score_torch(mask1, mask2, smooth=1e-8):
    """
    Compute Dice Score between two binary masks.
    Args:
        mask1, mask2: Binary masks (torch tensors or numpy arrays)
        smooth: Smoothing factor to avoid division by zero
    Returns:
        Dice score (float): Higher is better
    """
    if not torch.is_tensor(mask1):
        mask1 = torch.from_numpy(mask1).float()
    if not torch.is_tensor(mask2):
        mask2 = torch.from_numpy(mask2).float()
    
    # Ensure masks are flattened for computation
    mask1_flat = mask1.view(-1)
    mask2_flat = mask2.view(-1)
    
    intersection = torch.sum(mask1_flat * mask2_flat)
    union = torch.sum(mask1_flat) + torch.sum(mask2_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def compute_hu_moments(image_np):
    """
    Compute Hu Moments for shape description.
    Args:
        image_np: Grayscale image as numpy array
    Returns:
        hu_moments: Array of 7 Hu moment invariants
    """
    try:
        import cv2
        
        # Ensure image is in correct format
        if image_np.dtype != np.uint8:
            image_np = (255 * (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-8)).astype(np.uint8)
        
        # Compute moments
        moments = cv2.moments(image_np)
        
        # Compute Hu moments
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Apply log transformation to make them more comparable
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
        
        return hu_moments
    
    except ImportError:
        # Fallback: simple geometric moments if OpenCV not available
        st.warning("⚠️ OpenCV not available. Using simplified moments.")
        
        # Create binary mask
        binary_mask = image_np > np.mean(image_np)
        
        if np.sum(binary_mask) == 0:
            return np.zeros(7)
        
        # Simple centroid and basic shape descriptors
        y_coords, x_coords = np.where(binary_mask)
        
        if len(x_coords) == 0:
            return np.zeros(7)
        
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        
        # Basic shape descriptors
        area = len(x_coords)
        perimeter = np.sum(binary_mask) - np.sum(binary_mask[1:-1, 1:-1])
        
        # Normalized moments (simplified)
        m20 = np.sum((x_coords - centroid_x) ** 2) / area
        m02 = np.sum((y_coords - centroid_y) ** 2) / area
        m11 = np.sum((x_coords - centroid_x) * (y_coords - centroid_y)) / area
        
        # Simple invariants
        eccentricity = (m20 + m02 + np.sqrt((m20 - m02)**2 + 4*m11**2)) / (m20 + m02 - np.sqrt((m20 - m02)**2 + 4*m11**2) + 1e-8)
        
        return np.array([area/1000, perimeter/100, eccentricity, m20/1000, m02/1000, m11/1000, 0])

def hu_moments_distance(hu1, hu2):
    """
    Compute distance between two sets of Hu moments.
    Lower distance indicates better match.
    """
    if len(hu1) != len(hu2):
        return float('inf')
    
    # Use Euclidean distance
    distance = np.sqrt(np.sum((hu1 - hu2) ** 2))
    return distance

def coarse_search_score(dice_val, hu_distance, w_dice=0.4, w_hu=0.6, max_hu_distance=10.0):
    """
    Compute coarse search score combining Dice and Hu moments.
    
    IMPROVED: Increased Hu moments weight from 0.3 to 0.6 for better rotation invariance.
    The Hu Moments are theoretically rotation-invariant, so giving them more weight
    improves shape similarity recognition over simple overlap.
    
    Args:
        dice_val: Dice score (higher is better)
        hu_distance: Hu moments distance (lower is better)  
        w_dice, w_hu: Weights for Dice and Hu components (now favors Hu moments)
        max_hu_distance: Maximum expected Hu distance for normalization
    Returns:
        Combined coarse score (higher is better)
    """
    # Convert Hu distance to similarity (higher is better)
    hu_similarity = max(0, 1.0 - (hu_distance / max_hu_distance))
    
    # Combine weighted scores - now emphasizing shape similarity via Hu moments
    coarse_score = w_dice * dice_val + w_hu * hu_similarity
    
    return coarse_score

def orb_feature_matching(img1, img2, max_features=500, match_threshold=0.75):
    """
    Compute ORB (Oriented FAST and Rotated BRIEF) feature matching score between two images.
    
    ORB is a fast, effective, and patent-free feature detection algorithm that combines:
    - FAST keypoint detector 
    - BRIEF descriptor with orientation compensation
    - Rotation invariance and scale invariance
    
    Args:
        img1: First image (numpy array, grayscale)
        img2: Second image (numpy array, grayscale) 
        max_features: Maximum number of features to detect
        match_threshold: Distance ratio threshold for good matches (Lowe's ratio)
    
    Returns:
        orb_score: Feature matching score [0,1] (higher is better)
        num_matches: Number of good feature matches found
    """
    try:
        # Convert to uint8 if needed (ORB requires 8-bit images)
        if img1.dtype != np.uint8:
            img1 = ((img1 - img1.min()) / (img1.max() - img1.min() + 1e-8) * 255).astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = ((img2 - img2.min()) / (img2.max() - img2.min() + 1e-8) * 255).astype(np.uint8)
            
        # Initialize ORB detector with optimized parameters for fossil matching
        orb = cv2.ORB_create(
            nfeatures=max_features,      # Maximum number of features to retain
            scaleFactor=1.2,             # Pyramid decimation ratio
            nlevels=8,                   # Number of pyramid levels
            edgeThreshold=31,            # Size of border where features are not detected
            firstLevel=0,                # Level of pyramid to put source image to
            WTA_K=2,                     # Number of points that produce each element of descriptor
            scoreType=cv2.ORB_HARRIS_SCORE,  # Harris corner detector for better corner response
            patchSize=31,                # Size of patch used for oriented BRIEF descriptor
            fastThreshold=20             # Fast threshold for corner detection
        )
        
        # Detect keypoints and compute descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        # Check if descriptors were found
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            return 0.0, 0
            
        # Create BFMatcher (Brute Force Matcher) with Hamming distance for binary descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Find k=2 best matches for each descriptor (for ratio test)
        if len(des1) >= 2 and len(des2) >= 2:
            matches = bf.knnMatch(des1, des2, k=2)
        else:
            # Fallback for cases with very few features
            matches = bf.match(des1, des2)
            # Convert to knnMatch format
            matches = [[m] for m in matches]
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Good match if distance ratio is below threshold
                if m.distance < match_threshold * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                # Only one match found, accept it (fallback case)
                good_matches.append(match_pair[0])
        
        num_matches = len(good_matches)
        
        # Compute feature matching score
        if num_matches == 0:
            orb_score = 0.0
        else:
            # Normalize by total possible matches (considering both images' features)
            max_possible_matches = min(len(kp1), len(kp2))
            if max_possible_matches > 0:
                # Base score from match ratio
                match_ratio = num_matches / max_possible_matches
                
                # Bonus for high absolute number of matches (indicates rich feature content)
                match_bonus = min(1.0, num_matches / 50.0)  # Bonus caps at 50 matches
                
                # Combined score with diminishing returns
                orb_score = min(1.0, match_ratio * 0.7 + match_bonus * 0.3)
            else:
                orb_score = 0.0
                
        return orb_score, num_matches
        
    except Exception as e:
        # Graceful fallback if ORB fails - show error in UI for debugging
        st.write(f"    ⚠️ ORB feature matching error: {str(e)}")
        return 0.0, 0

def fine_tuning_score(dice_val, ncc_val, ssim_val=None, orb_val=None, 
                     w_dice_fine=0.3, w_ncc_fine=0.25, w_ssim_fine=0.25, w_orb_fine=0.2):
    """
    Compute fine-tuning score combining Dice, NCC, SSIM, and ORB for enhanced structural sensitivity.
    
    ENHANCED: Now includes ORB feature matching for rotation-invariant structural analysis.
    
    Args:
        dice_val: Dice score on masks (higher is better, [0,1])
        ncc_val: NCC on grayscale images ([-1,1], normalized to [0,1]) 
        ssim_val: SSIM on grayscale images ([0,1], higher is better)
        orb_val: ORB feature matching score ([0,1], higher is better)
        w_dice_fine: Weight for Dice component (shape overlap)
        w_ncc_fine: Weight for NCC component (intensity correlation)
        w_ssim_fine: Weight for SSIM component (structural similarity)
        w_orb_fine: Weight for ORB component (feature-based matching)
    Returns:
        Combined fine-tuning score (higher is better, [0,1])
    """
    # Normalize NCC from [-1,1] to [0,1]
    ncc_normalized = (ncc_val + 1.0) / 2.0
    
    # Check which metrics are available and compute weighted combination
    if orb_val is not None and ssim_val is not None:
        # Four-metric combination: Dice + NCC + SSIM + ORB
        total_weight = w_dice_fine + w_ncc_fine + w_ssim_fine + w_orb_fine
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights if they don't sum to 1
            w_dice_fine /= total_weight
            w_ncc_fine /= total_weight  
            w_ssim_fine /= total_weight
            w_orb_fine /= total_weight
        
        # Four-metric weighted combination
        fine_score = (w_dice_fine * dice_val + 
                     w_ncc_fine * ncc_normalized + 
                     w_ssim_fine * ssim_val +
                     w_orb_fine * orb_val)
                     
    elif ssim_val is not None:
        # Three-metric combination: Dice + NCC + SSIM (legacy mode)
        total_weight = w_dice_fine + w_ncc_fine + w_ssim_fine
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights if they don't sum to 1
            w_dice_fine /= total_weight
            w_ncc_fine /= total_weight
            w_ssim_fine /= total_weight
        
        # Three-metric weighted combination
        fine_score = (w_dice_fine * dice_val + 
                     w_ncc_fine * ncc_normalized + 
                     w_ssim_fine * ssim_val)
    else:
        # Fallback to original two-metric combination
        # Adjust weights to sum to 1.0
        total_weight = w_dice_fine + w_ncc_fine
        if abs(total_weight - 1.0) > 1e-6:
            w_dice_fine /= total_weight
            w_ncc_fine /= total_weight
        
        fine_score = w_dice_fine * dice_val + w_ncc_fine * ncc_normalized
    
    return fine_score

def create_binary_mask(image, threshold_method='otsu'):
    """
    Create binary mask from grayscale image.
    Args:
        image: Input grayscale image (numpy array or torch tensor)
        threshold_method: 'otsu', 'adaptive', or 'percentile'
    Returns:
        Binary mask as numpy array
    """
    try:
        from skimage import filters
        
        # Convert to numpy if needed
        if torch.is_tensor(image):
            img_np = image.cpu().numpy()
        else:
            img_np = image.copy()
        
        # Ensure 2D
        if img_np.ndim > 2:
            img_np = img_np.squeeze()
        
        if threshold_method == 'otsu':
            try:
                threshold = filters.threshold_otsu(img_np)
                binary_mask = img_np > threshold
            except ValueError:
                # Fallback to percentile if Otsu fails
                threshold = np.percentile(img_np, 75)
                binary_mask = img_np > threshold
        
        elif threshold_method == 'adaptive':
            # Use mean + std approach
            threshold = np.mean(img_np) + 0.5 * np.std(img_np)
            binary_mask = img_np > threshold
        
        elif threshold_method == 'percentile':
            threshold = np.percentile(img_np, 70)
            binary_mask = img_np > threshold
        
        else:
            # Default to simple mean threshold
            threshold = np.mean(img_np)
            binary_mask = img_np > threshold
        
        return binary_mask.astype(np.float32)
    
    except ImportError:
        # Fallback without scikit-image
        if torch.is_tensor(image):
            img_np = image.cpu().numpy()
        else:
            img_np = image.copy()
        
        if img_np.ndim > 2:
            img_np = img_np.squeeze()
        
        threshold = np.mean(img_np) + 0.5 * np.std(img_np)
        binary_mask = (img_np > threshold).astype(np.float32)
        
        return binary_mask

def stage1_coarse_search(volume_t, slice_t, axes=[0,1,2], fossil_area_threshold=0.05, 
                        use_segmentation=False, segmentation_threshold=0.5):
    """
    Stage 1: Coarse Search for Initial Pose Estimation using Dice + Hu Moments.
    
    Args:
        volume_t: 4D torch tensor (1,1,D,H,W) - the 3D volume
        slice_t: 4D torch tensor (1,1,H,W) - the query slice
        axes: List of axes to search (0=axial, 1=coronal, 2=sagittal)
        fossil_area_threshold: Minimum fossil content ratio for valid slices
        use_segmentation: Whether to apply advanced segmentation to volume slices
        segmentation_threshold: Threshold for segmentation (if enabled)
    
    Returns:
        initial_pose: Dict containing best (axis, index) and score
    """
    best_coarse = {"score": -1, "axis": None, "index": None, "dice": 0, "hu_distance": float('inf')}
    
    D, H, W = volume_t.shape[2:]
    h_s, w_s = slice_t.shape[2:]
    
    # Convert volume to numpy for mask calculation
    vol_np = volume_t.cpu().numpy()[0, 0]
    
    # Build fossil mask for valid slice detection
    thr = smart_threshold(vol_np)
    try:
        mask, bbox = build_fossil_mask(vol_np, thr)
    except (ValueError, IndexError) as e:
        st.warning(f"⚠️ Stage 1: Could not detect fossil content in volume: {e}")
        return best_coarse
    
    # Create binary mask for query slice
    slice_np = slice_t.cpu().numpy()[0, 0]
    if use_segmentation:
        # Apply the same advanced segmentation as used for uploaded image
        segmented_query_slice = apply_fossil_segmentation(slice_np, segmentation_threshold, 
                                                          show_preview=False, preserve_holes=preserve_holes)
        query_mask = create_binary_mask(segmented_query_slice, 'otsu')
        # st.write(f"  🎯 Using segmented query slice for Stage 1 coarse search")
    else:
        query_mask = create_binary_mask(slice_np, 'otsu')
    
    # Compute Hu moments for query slice (use segmented version if available)
    query_slice_for_hu = segmented_query_slice if use_segmentation else slice_np
    query_hu = compute_hu_moments(query_slice_for_hu)
    
    total_slices_checked = 0
    valid_slices_found = 0
    
    for axis in axes:
        # Get valid indices with fossil content
        valid_indices = valid_idx(mask, axis, bbox)
        
        if len(valid_indices) == 0:
            continue
        
        axis_name = ["axial", "coronal", "sagittal"][axis]
        
        for idx in valid_indices:
            total_slices_checked += 1
            
            # Extract slice from volume
            if axis == 0:  # Axial
                vol_slice = volume_t[:, :, idx, :, :]
            elif axis == 1:  # Coronal  
                vol_slice = volume_t[:, :, :, idx, :].permute(0, 1, 3, 2)
            else:  # Sagittal
                vol_slice = volume_t[:, :, :, :, idx]
            
            # Resize to match query slice
            if vol_slice.shape[2:] != (h_s, w_s):
                vol_slice = F.interpolate(vol_slice, size=(h_s, w_s), mode='bilinear', align_corners=False)
            
            # Apply segmentation to volume slice if enabled (for consistency)
            if use_segmentation:
                vol_slice_segmented = segment_volume_slice(vol_slice, segmentation_threshold, preserve_holes)
                vol_slice_np = vol_slice_segmented.cpu().numpy()[0, 0]
            else:
                vol_slice_np = vol_slice.cpu().numpy()[0, 0]
            
            # Create binary mask for volume slice
            vol_mask = create_binary_mask(vol_slice_np, 'otsu')
            
            # Check if volume slice has reasonable fossil content
            if np.sum(vol_mask) < vol_mask.size * 0.01:  # Less than 1% fossil content
                continue
                
            valid_slices_found += 1
            
            # Compute Dice score
            dice_val = dice_score_torch(query_mask, vol_mask)
            
            # Compute Hu moments and distance (use segmented version for consistency)
            vol_hu = compute_hu_moments(vol_slice_np)
            hu_dist = hu_moments_distance(query_hu, vol_hu)
            
            # Compute coarse score
            coarse_score = coarse_search_score(dice_val, hu_dist)
            
            if coarse_score > best_coarse["score"]:
                best_coarse.update({
                    "score": coarse_score,
                    "axis": axis,
                    "index": idx,
                    "dice": dice_val,
                    "hu_distance": hu_dist,
                    "slice_2d": vol_slice_segmented if use_segmentation else vol_slice.clone()
                })
    
    # Debug output
    seg_status = "with segmentation" if use_segmentation else "without segmentation"
    st.write(f"  📊 Checked {total_slices_checked} slices, {valid_slices_found} had sufficient fossil content ({seg_status})")
    if best_coarse["score"] > 0:
        axis_name = ["axial", "coronal", "sagittal"][best_coarse["axis"]]
        st.write(f"  🎯 Best coarse match: {axis_name} axis, slice {best_coarse['index']}")
        st.write(f"  📈 Dice: {best_coarse['dice']:.4f}, Hu distance: {best_coarse['hu_distance']:.4f}")
    
    return best_coarse

def rotate_image_torch(image_tensor, angle_degrees):
    """
    Rotate a 2D image tensor by the specified angle in degrees.
    
    Args:
        image_tensor: 4D torch tensor (1,1,H,W) or 2D tensor (H,W)
        angle_degrees: Rotation angle in degrees (positive = counter-clockwise)
    
    Returns:
        Rotated image tensor with same dimensions as input
    """
    import torch
    import torch.nn.functional as F
    
    # Ensure input is 4D
    original_shape = image_tensor.shape
    if len(original_shape) == 2:
        image_4d = image_tensor.unsqueeze(0).unsqueeze(0)
    elif len(original_shape) == 3:
        image_4d = image_tensor.unsqueeze(0)
    else:
        image_4d = image_tensor
    
    # Convert angle to radians
    angle_rad = torch.tensor(angle_degrees * np.pi / 180.0, dtype=torch.float32, device=image_tensor.device)
    
    # Create rotation matrix
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    
    # Rotation matrix for counter-clockwise rotation
    rotation_matrix = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=torch.float32, device=image_tensor.device).unsqueeze(0)
    
    # Create affine grid
    _, _, H, W = image_4d.shape
    grid = F.affine_grid(rotation_matrix, (1, 1, H, W), align_corners=False)
    
    # Apply rotation
    rotated = F.grid_sample(image_4d, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    # Return in original shape format
    if len(original_shape) == 2:
        return rotated.squeeze()
    elif len(original_shape) == 3:
        return rotated.squeeze(0)
    else:
        return rotated


def stage2_fine_tuning_with_rotation(volume_t, slice_t, initial_pose, search_radius=5, 
                                   use_segmentation=False, segmentation_threshold=0.5,
                                   rotation_angles=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                                   preserve_holes=True,
                                   w_dice_fine=0.3, w_ncc_fine=0.25, w_ssim_fine=0.25, w_orb_fine=0.2):
    """
    Enhanced Stage 2: Fine-Tuning with In-Plane Rotation Testing for Rotation Invariance.
    
    This function implements the BEST FIX suggested: Add In-Plane Rotation to the Fine-Tuning Stage.
    For each candidate slice from the 3D volume, it tests multiple rotations and uses the best score.
    
    Args:
        volume_t: 4D torch tensor (1,1,D,H,W) - the 3D volume
        slice_t: 4D torch tensor (1,1,H,W) - the query slice
        initial_pose: Dict from Stage 1 with initial (axis, index)
        search_radius: Number of slices around initial pose to search
        use_segmentation: Whether to apply advanced segmentation
        segmentation_threshold: Threshold for segmentation (if enabled)
        rotation_angles: List of angles to test for each candidate slice
    
    Returns:
        final_pose: Dict containing optimized pose, final score, and best rotation
    """
    if initial_pose["axis"] is None or initial_pose["index"] is None:
        return {"score": 0, "final_pose": None, "dice": 0, "ncc": 0, "ssim": 0, "orb": 0, "orb_matches": 0, "best_rotation": 0}
    
    best_fine = {"score": -1, "axis": None, "index": None, "dice": 0, "ncc": 0, "ssim": 0, "orb": 0, "orb_matches": 0, "best_rotation": 0}
    
    axis = initial_pose["axis"]
    center_idx = initial_pose["index"]
    D, H, W = volume_t.shape[2:]
    h_s, w_s = slice_t.shape[2:]
    
    # Determine search range around initial pose
    if axis == 0:  # Axial
        max_idx = D - 1
    elif axis == 1:  # Coronal
        max_idx = H - 1
    else:  # Sagittal
        max_idx = W - 1
    
    start_idx = max(0, center_idx - search_radius)
    end_idx = min(max_idx, center_idx + search_radius)
    
    # Create masks for query slice (use segmented version if enabled)
    slice_np = slice_t.cpu().numpy()[0, 0]
    if use_segmentation:
        segmented_query_slice = apply_fossil_segmentation(slice_np, segmentation_threshold, 
                                                          show_preview=False, preserve_holes=preserve_holes)
        query_mask = create_binary_mask(segmented_query_slice, 'otsu')
        # Use segmented version for NCC comparison too
        query_slice_for_ncc = torch.from_numpy(segmented_query_slice[None, None]).float().to(slice_t.device)
        query_slice_for_ncc = normalize_to_01_torch(query_slice_for_ncc)
        # st.write(f"  🎯 Using segmented query slice for Stage 2 fine-tuning with rotation")
    else:
        query_mask = create_binary_mask(slice_np, 'otsu')
        query_slice_for_ncc = slice_t
    
    # Search around initial pose with fine-tuning metric and rotation testing
    slices_refined = 0
    total_rotations_tested = 0
    
    # st.write(f"  🔄 Testing {len(rotation_angles)} rotations per slice for rotation invariance")
    
    for idx in range(start_idx, end_idx + 1):
        slices_refined += 1
        
        # Extract slice from volume
        if axis == 0:  # Axial
            vol_slice = volume_t[:, :, idx, :, :]
        elif axis == 1:  # Coronal
            vol_slice = volume_t[:, :, :, idx, :].permute(0, 1, 3, 2)
        else:  # Sagittal
            vol_slice = volume_t[:, :, :, :, idx]
        
        # Resize to match query slice
        if vol_slice.shape[2:] != (h_s, w_s):
            vol_slice = F.interpolate(vol_slice, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
        # Apply segmentation to volume slice if enabled (for consistency)
        if use_segmentation:
            vol_slice_segmented = segment_volume_slice(vol_slice, segmentation_threshold, preserve_holes)
            vol_slice_np = vol_slice_segmented.cpu().numpy()[0, 0]
            vol_slice_for_rotation = vol_slice_segmented  # Use segmented version for rotation
        else:
            vol_slice_np = vol_slice.cpu().numpy()[0, 0]
            vol_slice_for_rotation = vol_slice
        
        # NEW: Test multiple rotations of the candidate slice
        best_rotation_score = -1
        best_rotation_angle = 0
        best_rotation_dice = 0
        best_rotation_ncc = 0
        best_rotation_ssim = 0  # Track SSIM
        best_rotation_orb = 0   # NEW: Track ORB
        best_rotation_matches = 0  # NEW: Track ORB match count
        best_rotated_slice = None
        
        for angle in rotation_angles:
            total_rotations_tested += 1
            
            # Rotate the candidate slice from the volume
            if angle == 0:
                # No rotation needed
                rotated_vol_slice = vol_slice_for_rotation
            else:
                # Apply rotation
                rotated_vol_slice = rotate_image_torch(vol_slice_for_rotation, angle)
            
            # Create binary mask for rotated volume slice
            rotated_vol_slice_np = rotated_vol_slice.cpu().numpy()[0, 0]
            rotated_vol_mask = create_binary_mask(rotated_vol_slice_np, 'otsu')
            
            # Compute Dice score on masks
            dice_val = dice_score_torch(query_mask, rotated_vol_mask)
            
            # Compute NCC on grayscale images
            ncc_val = ncc_torch(query_slice_for_ncc, rotated_vol_slice)
            
            # Compute SSIM for structural similarity
            try:
                ssim_val = ssim_torch(query_slice_for_ncc, rotated_vol_slice)
                if ssim_val is None or ssim_val == 0:
                    st.write(f"    ⚠️ Debug: SSIM is {ssim_val} for angle {angle}°")
            except Exception as e:
                st.write(f"    ❌ SSIM computation failed for angle {angle}°: {e}")
                ssim_val = 0.0
            
            # NEW: Compute ORB feature matching for rotation-invariant structural analysis
            try:
                query_slice_np = query_slice_for_ncc.cpu().numpy()[0, 0]
                orb_score, num_matches = orb_feature_matching(query_slice_np, rotated_vol_slice_np)
                # Debug: Show ORB computation for first few angles
                if total_rotations_tested <= 3:
                    # st.write(f"    🔍 ORB Debug - Angle {angle}°: Score={orb_score:.4f}, Matches={num_matches}")
                    pass
            except Exception as e:
                st.write(f"    ❌ ORB feature matching failed for angle {angle}°: {e}")
                orb_score = 0.0
                num_matches = 0
            
            # Compute enhanced fine-tuning score with four metrics (Dice + NCC + SSIM + ORB)
            fine_score = fine_tuning_score(dice_val, ncc_val, ssim_val, orb_score,
                                         w_dice_fine=w_dice_fine, w_ncc_fine=w_ncc_fine, w_ssim_fine=w_ssim_fine, w_orb_fine=w_orb_fine)
            
            # Track the best rotation for this slice
            if fine_score > best_rotation_score:
                best_rotation_score = fine_score
                best_rotation_angle = angle
                best_rotation_dice = dice_val
                best_rotation_ncc = ncc_val
                best_rotation_ssim = ssim_val  # Track best SSIM
                best_rotation_orb = orb_score   # NEW: Track best ORB
                best_rotation_matches = num_matches  # NEW: Track ORB matches
                best_rotated_slice = rotated_vol_slice.clone()
        
        # Update global best if this slice (with its best rotation) is better
        if best_rotation_score > best_fine["score"]:
            best_fine.update({
                "score": best_rotation_score,
                "axis": axis,
                "index": idx,
                "dice": best_rotation_dice,
                "ncc": best_rotation_ncc,
                "ssim": best_rotation_ssim,  # Include SSIM in results
                "orb": best_rotation_orb,    # NEW: Include ORB in results
                "orb_matches": best_rotation_matches,  # NEW: Include ORB match count
                "best_rotation": best_rotation_angle,
                "slice_2d": best_rotated_slice,
                "final_pose": {"axis": axis, "index": idx, "rotation": best_rotation_angle}
            })
    
    # Debug output with rotation information
    axis_name = ["axial", "coronal", "sagittal"][axis]
    seg_status = "with segmentation" if use_segmentation else "without segmentation"
    # st.write(f"  🔧 Refined {slices_refined} slices around {axis_name} slice {center_idx} ({seg_status})")
    # st.write(f"  🔄 Tested {total_rotations_tested} total rotations ({len(rotation_angles)} per slice)")
    
    if best_fine["score"] > 0:
        improvement = best_fine["score"] - (initial_pose.get("score", 0) if "score" in initial_pose else 0)
        # st.write(f"  ⬆️ Final slice {best_fine['index']} with {best_fine['best_rotation']}° rotation")
        
        # Display all four metrics in the score improvement message (commented out for cleaner UI)
        # orb_info = f", ORB: {best_fine.get('orb', 0):.4f}" if 'orb' in best_fine else ""
        # st.write(f"  📈 Score improved by {improvement:.4f} - Dice: {best_fine['dice']:.4f}, NCC: {best_fine['ncc']:.4f}, SSIM: {best_fine['ssim']:.4f}{orb_info}")
        
        # Show ORB feature matches if available (commented out for cleaner UI)
        # if 'orb_matches' in best_fine and best_fine.get('orb_matches', 0) > 0:
        #     st.write(f"  🔍 ORB Feature Matches: {best_fine['orb_matches']}")
        
        # if best_fine['best_rotation'] != 0:
        #     st.success(f"  🎯 **Rotation Correction Applied:** {best_fine['best_rotation']}° improved the match!")
    
    return best_fine


def stage2_fine_tuning(volume_t, slice_t, initial_pose, search_radius=5, 
                      use_segmentation=False, segmentation_threshold=0.5, preserve_holes=True):
    """
    Stage 2: Fine-Tuning with 2D-to-3D Registration using Dice + NCC + SSIM + ORB.
    This is the legacy implementation - updated to include ORB feature matching.
    
    ENHANCED: Now includes ORB feature matching for better structural analysis.
    
    Args:
        volume_t: 4D torch tensor (1,1,D,H,W) - the 3D volume
        slice_t: 4D torch tensor (1,1,H,W) - the query slice
        initial_pose: Dict from Stage 1 with initial (axis, index)
        search_radius: Number of slices around initial pose to search
        use_segmentation: Whether to apply advanced segmentation
        segmentation_threshold: Threshold for segmentation (if enabled)
    
    Returns:
        final_pose: Dict containing optimized pose and final score
    """
    if initial_pose["axis"] is None or initial_pose["index"] is None:
        return {"score": 0, "final_pose": None, "dice": 0, "ncc": 0, "ssim": 0, "orb": 0, "orb_matches": 0}
    
    best_fine = {"score": -1, "axis": None, "index": None, "dice": 0, "ncc": 0, "ssim": 0, "orb": 0, "orb_matches": 0}
    
    axis = initial_pose["axis"]
    center_idx = initial_pose["index"]
    D, H, W = volume_t.shape[2:]
    h_s, w_s = slice_t.shape[2:]
    
    # Determine search range around initial pose
    if axis == 0:  # Axial
        max_idx = D - 1
    elif axis == 1:  # Coronal
        max_idx = H - 1
    else:  # Sagittal
        max_idx = W - 1
    
    start_idx = max(0, center_idx - search_radius)
    end_idx = min(max_idx, center_idx + search_radius)
    
    # Create masks for query slice (use segmented version if enabled)
    slice_np = slice_t.cpu().numpy()[0, 0]
    if use_segmentation:
        segmented_query_slice = apply_fossil_segmentation(slice_np, segmentation_threshold, 
                                                          show_preview=False, preserve_holes=preserve_holes)
        query_mask = create_binary_mask(segmented_query_slice, 'otsu')
        # Use segmented version for NCC comparison too
        query_slice_for_ncc = torch.from_numpy(segmented_query_slice[None, None]).float().to(slice_t.device)
        query_slice_for_ncc = normalize_to_01_torch(query_slice_for_ncc)
        # st.write(f"  🎯 Using segmented query slice for Stage 2 fine-tuning")
    else:
        query_mask = create_binary_mask(slice_np, 'otsu')
        query_slice_for_ncc = slice_t
    
    # Search around initial pose with fine-tuning metric
    slices_refined = 0
    
    for idx in range(start_idx, end_idx + 1):
        slices_refined += 1
        
        # Extract slice from volume
        if axis == 0:  # Axial
            vol_slice = volume_t[:, :, idx, :, :]
        elif axis == 1:  # Coronal
            vol_slice = volume_t[:, :, :, idx, :].permute(0, 1, 3, 2)
        else:  # Sagittal
            vol_slice = volume_t[:, :, :, :, idx]
        
        # Resize to match query slice
        if vol_slice.shape[2:] != (h_s, w_s):
            vol_slice = F.interpolate(vol_slice, size=(h_s, w_s), mode='bilinear', align_corners=False)
        
        # Apply segmentation to volume slice if enabled (for consistency)
        if use_segmentation:
            vol_slice_segmented = segment_volume_slice(vol_slice, segmentation_threshold, preserve_holes)
            vol_slice_np = vol_slice_segmented.cpu().numpy()[0, 0]
            vol_slice_for_ncc = vol_slice_segmented  # Use segmented version for NCC too
        else:
            vol_slice_np = vol_slice.cpu().numpy()[0, 0]
            vol_slice_for_ncc = vol_slice
        
        # Create binary mask for volume slice
        vol_mask = create_binary_mask(vol_slice_np, 'otsu')
        
        # Compute Dice score on masks
        dice_val = dice_score_torch(query_mask, vol_mask)
        
        # Compute NCC on grayscale images (both segmented if segmentation enabled)
        ncc_val = ncc_torch(query_slice_for_ncc, vol_slice_for_ncc)
        
        # Compute SSIM for structural similarity
        try:
            ssim_val = ssim_torch(query_slice_for_ncc, vol_slice_for_ncc)
            if ssim_val is None or ssim_val == 0:
                st.write(f"    ⚠️ Debug: SSIM is {ssim_val} for slice {idx}")
        except Exception as e:
            st.write(f"    ❌ SSIM computation failed for slice {idx}: {e}")
            ssim_val = 0.0
            
        # NEW: Compute ORB feature matching for structural analysis  
        try:
            query_slice_np = query_slice_for_ncc.cpu().numpy()[0, 0]
            orb_score, num_matches = orb_feature_matching(query_slice_np, vol_slice_np)
            # Debug: Show ORB computation for first few slices
            if slices_refined <= 3:
                # st.write(f"    🔍 ORB Debug - Slice {idx}: Score={orb_score:.4f}, Matches={num_matches}")
                pass
        except Exception as e:
            st.write(f"    ❌ ORB feature matching failed for slice {idx}: {e}")
            orb_score = 0.0
            num_matches = 0
        
        # Compute enhanced fine-tuning score with four metrics (Dice + NCC + SSIM + ORB)
        fine_score = fine_tuning_score(dice_val, ncc_val, ssim_val, orb_score,
                                     w_dice_fine=w_dice_fine, w_ncc_fine=w_ncc_fine, w_ssim_fine=w_ssim_fine, w_orb_fine=w_orb_fine)
        
        if fine_score > best_fine["score"]:
            best_fine.update({
                "score": fine_score,
                "axis": axis,
                "index": idx,
                "dice": dice_val,
                "ncc": ncc_val,
                "ssim": ssim_val,  # Include SSIM in results
                "orb": orb_score,   # NEW: Include ORB in results
                "orb_matches": num_matches,  # NEW: Include ORB match count
                "slice_2d": vol_slice_segmented if use_segmentation else vol_slice.clone(),
                "final_pose": {"axis": axis, "index": idx}
            })
    
    # Debug output
    axis_name = ["axial", "coronal", "sagittal"][axis]
    seg_status = "with segmentation" if use_segmentation else "without segmentation"
    # st.write(f"  🔧 Refined {slices_refined} slices around {axis_name} slice {center_idx} ({seg_status})")
    if best_fine["score"] > 0:
        improvement = best_fine["score"] - (initial_pose.get("score", 0) if "score" in initial_pose else 0)
        # st.write(f"  ⬆️ Final slice {best_fine['index']}, score improved by {improvement:.4f}")
        
        # Display all four metrics including ORB (commented out for cleaner UI)
        # orb_info = f", ORB: {best_fine.get('orb', 0):.4f}" if 'orb' in best_fine else ""
        # st.write(f"  📈 Final metrics - Dice: {best_fine['dice']:.4f}, NCC: {best_fine['ncc']:.4f}, SSIM: {best_fine['ssim']:.4f}{orb_info}")
        
        # Show ORB feature matches if available (commented out for cleaner UI)
        # if 'orb_matches' in best_fine and best_fine.get('orb_matches', 0) > 0:
        #     st.write(f"  🔍 ORB Feature Matches: {best_fine['orb_matches']}")
        pass
    
    return best_fine

def stage2_fine_tuning_diagonal_candidate(volume_t, slice_t, diagonal_candidate, 
                                        use_segmentation=False, segmentation_threshold=0.5,
                                        preserve_holes=True,
                                        w_dice_fine=0.3, w_ncc_fine=0.25, w_ssim_fine=0.25, w_orb_fine=0.2):
    """
    Stage 2 fine-tuning specifically for diagonal candidates.
    Since diagonal slices are already optimal, we apply rotation testing to them.
    """
    # Use the diagonal slice directly from the candidate
    diagonal_slice = diagonal_candidate["slice_2d"]
    
    # Create masks for query slice (use segmented version if enabled)
    slice_np = slice_t.cpu().numpy()[0, 0]
    if use_segmentation:
        segmented_query_slice = apply_fossil_segmentation(slice_np, segmentation_threshold, 
                                                          show_preview=False, preserve_holes=preserve_holes)
        query_mask = create_binary_mask(segmented_query_slice, 'otsu')
        query_slice_for_ncc = torch.from_numpy(segmented_query_slice[None, None]).float().to(slice_t.device)
        query_slice_for_ncc = normalize_to_01_torch(query_slice_for_ncc)
    else:
        query_mask = create_binary_mask(slice_np, 'otsu')
        query_slice_for_ncc = slice_t
    
    # Test rotations on the diagonal slice
    rotation_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    best_result = {"score": -1}
    
    for angle in rotation_angles:
        try:
            # Rotate the diagonal slice
            rotated_slice = rotate_tensor_2d(diagonal_slice, angle)
            
            # Compute all four metrics
            rotated_slice_np = rotated_slice.cpu().numpy()[0, 0]
            
            # Dice coefficient
            rotated_mask = create_binary_mask(rotated_slice_np, 'otsu')
            dice_val = dice_coefficient(query_mask, rotated_mask)
            
            # NCC
            ncc_val = ncc_torch(query_slice_for_ncc, rotated_slice)
            ncc_val = (ncc_val + 1) / 2  # Normalize to [0, 1]
            
            # SSIM
            ssim_val = ssim_torch(query_slice_for_ncc, rotated_slice)
            ssim_val = (ssim_val + 1) / 2  # Normalize to [0, 1]
            
            # ORB feature matching
            query_slice_np = query_slice_for_ncc.cpu().numpy()[0, 0]
            orb_score, num_matches = orb_feature_matching(query_slice_np, rotated_slice_np)
            
            # Compute fine-tuning score
            fine_score = fine_tuning_score(dice_val, ncc_val, ssim_val, orb_score,
                                         w_dice_fine=w_dice_fine, w_ncc_fine=w_ncc_fine, 
                                         w_ssim_fine=w_ssim_fine, w_orb_fine=w_orb_fine)
            
            if fine_score > best_result["score"]:
                best_result = {
                    "score": fine_score,
                    "axis": "diagonal",
                    "angles": diagonal_candidate["angles"],
                    "rotation": angle,
                    "dice": dice_val,
                    "ncc": ncc_val,
                    "ssim": ssim_val,
                    "orb": orb_score,
                    "orb_matches": num_matches,
                    "slice_2d": rotated_slice.clone()
                }
                
        except Exception as e:
            continue
    
    return best_result

def multi_candidate_fine_tuning(volume_t, slice_t, axes=[0,1,2], fossil_area_threshold=0.05, 
                               search_radius=5, use_segmentation=False, segmentation_threshold=0.5,
                               enable_rotation_testing=True, preserve_holes=True, top_n_candidates=5,
                               w_dice_fine=0.3, w_ncc_fine=0.25, w_ssim_fine=0.25, w_orb_fine=0.2,
                               enable_diagonal=False, angle_list=None):
    """
    IMPROVED: Multi-Candidate Fine-Tuning Pipeline
    
    Conceptual Change:
    - Modify the main matching loop to first run Stage 1 for all selected models
    - Store the top N (e.g., top 5) initial poses based on their coarse_search_score
    - Then, in a second loop, run the powerful stage2_fine_tuning_with_rotation on each candidate
    - The final winner will be the candidate with the highest score after Stage 2
    
    This approach ensures that promising candidates aren't dismissed due to Stage 1 limitations,
    and allows the sophisticated Stage 2 to make the final choice among multiple good options.
    
    Args:
        volume_t: 4D torch tensor (1,1,D,H,W) - the 3D volume
        slice_t: 4D torch tensor (1,1,H,W) - the query slice
        axes: List of axes to search
        fossil_area_threshold: Minimum fossil content for valid slices
        search_radius: Fine-tuning search radius around coarse result
        use_segmentation: Whether to apply advanced segmentation consistently
        segmentation_threshold: Threshold for segmentation (if enabled)
        enable_rotation_testing: Whether to test rotations in Stage 2
        preserve_holes: Whether to preserve internal structure in segmentation
        top_n_candidates: Number of candidates to pass from Stage 1 to Stage 2
    
    Returns:
        final_result: Dict containing complete matching results with multi-candidate analysis
    """
    # Stage 1: Multi-Candidate Coarse Search
    if use_segmentation:
        # st.write(f"🔍 **Stage 1: Multi-Candidate Coarse Search** (Dice + Hu Moments) - Finding top {top_n_candidates} candidates with segmentation")
        pass
    else:
        # st.write(f"🔍 **Stage 1: Multi-Candidate Coarse Search** (Dice + Hu Moments) - Finding top {top_n_candidates} candidates")
        pass
    
    # Store ALL candidate poses from Stage 1
    all_candidates = []
    
    D, H, W = volume_t.shape[2:]
    h_s, w_s = slice_t.shape[2:]
    
    # Convert volume to numpy for mask calculation
    vol_np = volume_t.cpu().numpy()[0, 0]
    
    # Build fossil mask for valid slice detection
    thr = smart_threshold(vol_np)
    try:
        mask, bbox = build_fossil_mask(vol_np, thr)
    except (ValueError, IndexError) as e:
        st.warning(f"⚠️ Stage 1: Could not detect fossil content in volume: {e}")
        return {
            "mode": "multi_candidate_two_stage",
            "stage1_candidates": [],
            "stage2_results": [],
            "final_score": 0,
            "final_pose": None
        }
    
    # Create binary mask for query slice
    slice_np = slice_t.cpu().numpy()[0, 0]
    if use_segmentation:
        segmented_query_slice = apply_fossil_segmentation(slice_np, segmentation_threshold, 
                                                          show_preview=False, preserve_holes=preserve_holes)
        query_mask = create_binary_mask(segmented_query_slice, 'otsu')
    else:
        query_mask = create_binary_mask(slice_np, 'otsu')
    
    # Compute Hu moments for query slice
    query_slice_for_hu = segmented_query_slice if use_segmentation else slice_np
    query_hu = compute_hu_moments(query_slice_for_hu)
    
    total_slices_checked = 0
    valid_slices_found = 0
    
    # IMPROVED: Search ALL valid slices across ALL axes to find multiple candidates
    for axis in axes:
        valid_indices = valid_idx(mask, axis, bbox)
        
        if len(valid_indices) == 0:
            continue
        
        axis_name = ["axial", "coronal", "sagittal"][axis]
        
        for idx in valid_indices:
            total_slices_checked += 1
            
            # Extract slice from volume
            if axis == 0:  # Axial
                vol_slice = volume_t[:, :, idx, :, :]
            elif axis == 1:  # Coronal  
                vol_slice = volume_t[:, :, :, idx, :].permute(0, 1, 3, 2)
            else:  # Sagittal
                vol_slice = volume_t[:, :, :, :, idx]
            
            # Resize to match query slice
            if vol_slice.shape[2:] != (h_s, w_s):
                vol_slice = F.interpolate(vol_slice, size=(h_s, w_s), mode='bilinear', align_corners=False)
            
            # Apply segmentation to volume slice if enabled
            if use_segmentation:
                vol_slice_segmented = segment_volume_slice(vol_slice, segmentation_threshold, preserve_holes)
                vol_slice_np = vol_slice_segmented.cpu().numpy()[0, 0]
            else:
                vol_slice_np = vol_slice.cpu().numpy()[0, 0]
            
            # Create binary mask for volume slice
            vol_mask = create_binary_mask(vol_slice_np, 'otsu')
            
            # Check if volume slice has reasonable fossil content
            if np.sum(vol_mask) < vol_mask.size * 0.01:
                continue
                
            valid_slices_found += 1
            
            # Compute Dice score
            dice_val = dice_score_torch(query_mask, vol_mask)
            
            # Compute Hu moments and distance
            vol_hu = compute_hu_moments(vol_slice_np)
            hu_dist = hu_moments_distance(query_hu, vol_hu)
            
            # Compute coarse score
            coarse_score = coarse_search_score(dice_val, hu_dist)
            
            # Store this candidate for potential Stage 2 processing
            candidate = {
                "score": coarse_score,
                "axis": axis,
                "index": idx,
                "dice": dice_val,
                "hu_distance": hu_dist,
                "slice_2d": vol_slice_segmented if use_segmentation else vol_slice.clone(),
                "candidate_id": f"{axis_name}_{idx}"  # Unique identifier
            }
            
            all_candidates.append(candidate)
    
    # ENHANCED: Add diagonal search candidates if enabled
    if enable_diagonal and angle_list is not None:
        # st.write(f"🔍 **Enhanced Search**: Adding diagonal candidates using {len(angle_list)} optimized angles")
        
        D, H, W = volume_t.shape[2:]
        center = ((W-1)/2, (H-1)/2, (D-1)/2)
        out_shape = slice_t.shape[2:]
        
        diagonal_candidates_checked = 0
        
        # Use scientifically optimized angles for diagonal search
        for ax in angle_list:
            for ay in angle_list:
                for az in angle_list:
                    diagonal_candidates_checked += 1
                    
                    try:
                        # Extract diagonal slice
                        diagonal_slice = extract_arbitrary_slice_torch(volume_t, (ax, ay, az), center, out_shape)
                        
                        # Apply segmentation to diagonal slice if enabled
                        if use_segmentation:
                            diagonal_slice_segmented = segment_volume_slice(diagonal_slice, segmentation_threshold, preserve_holes)
                            diagonal_slice_np = diagonal_slice_segmented.cpu().numpy()[0, 0]
                        else:
                            diagonal_slice_np = diagonal_slice.cpu().numpy()[0, 0]
                        
                        # Check fossil content
                        diagonal_mask = create_binary_mask(diagonal_slice_np, 'otsu')
                        fossil_ratio = np.sum(diagonal_mask) / diagonal_mask.size
                        
                        if fossil_ratio < fossil_area_threshold:
                            continue
                        
                        valid_slices_found += 1
                        
                        # Compute Dice coefficient
                        dice_val = dice_coefficient(query_mask, diagonal_mask)
                        
                        # Compute Hu moments distance
                        diagonal_hu = compute_hu_moments(diagonal_slice_np)
                        hu_dist = np.sum(np.abs(np.array(query_hu) - np.array(diagonal_hu)))
                        hu_similarity = 1.0 / (1.0 + hu_dist)
                        
                        # Stage 1 coarse score (Dice + Hu Moments)
                        coarse_score = 0.7 * dice_val + 0.3 * hu_similarity
                        
                        # Store diagonal candidate
                        candidate = {
                            "score": coarse_score,
                            "axis": "diagonal",
                            "angles": (ax, ay, az),
                            "dice": dice_val,
                            "hu_distance": hu_dist,
                            "slice_2d": diagonal_slice_segmented if use_segmentation else diagonal_slice.clone(),
                            "candidate_id": f"diagonal_{ax}_{ay}_{az}"
                        }
                        
                        all_candidates.append(candidate)
                        
                    except Exception as e:
                        # Skip problematic diagonal slices
                        continue
        
        # st.write(f"  📐 Diagonal Search: Checked {diagonal_candidates_checked} angles, found {len([c for c in all_candidates if c.get('axis') == 'diagonal'])} valid diagonal candidates")
    
    # Sort all candidates by coarse score and select top N
    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = all_candidates[:top_n_candidates]
    
    # st.write(f"  📊 Stage 1 Complete: Checked {total_slices_checked} slices, found {valid_slices_found} with fossil content")
    # st.write(f"  🎯 Selected top {len(top_candidates)} candidates for Stage 2 fine-tuning:")
    
    # Comment out the candidate listing to reduce verbose output
    # for i, candidate in enumerate(top_candidates, 1):
    #     axis_name = ["axial", "coronal", "sagittal"][candidate["axis"]]
    #     st.write(f"    {i}. {axis_name} slice {candidate['index']} (score: {candidate['score']:.4f})")
    
    if not top_candidates:
        st.warning("⚠️ No valid candidates found in Stage 1")
        return {
            "mode": "multi_candidate_two_stage",
            "stage1_candidates": [],
            "stage2_results": [],
            "final_score": 0,
            "final_pose": None
        }
    
    # Stage 2: Fine-Tuning for Each Top Candidate
    if enable_rotation_testing:
        # st.write(f"🎯 **Stage 2: Multi-Candidate Fine-Tuning** - Processing {len(top_candidates)} candidates with rotation testing (Dice + NCC + SSIM + ORB)")
        pass
    else:
        # st.write(f"🎯 **Stage 2: Multi-Candidate Fine-Tuning** - Processing {len(top_candidates)} candidates (Dice + NCC + SSIM + ORB)")
        pass
    
    stage2_results = []
    
    for i, candidate in enumerate(top_candidates, 1):
        if candidate["axis"] == "diagonal":
            # st.write(f"  🔧 Processing candidate {i}/{len(top_candidates)}: diagonal angles {candidate['angles']}")
            
            # For diagonal candidates, use the candidate slice directly with rotation testing
            stage2_result = stage2_fine_tuning_diagonal_candidate(
                volume_t, slice_t, candidate, 
                use_segmentation, segmentation_threshold,
                preserve_holes=preserve_holes,
                w_dice_fine=w_dice_fine, w_ncc_fine=w_ncc_fine, w_ssim_fine=w_ssim_fine, w_orb_fine=w_orb_fine
            )
        else:
            axis_name = ["axial", "coronal", "sagittal"][candidate["axis"]]
            # st.write(f"  🔧 Processing candidate {i}/{len(top_candidates)}: {axis_name} slice {candidate['index']}")
            
            # Run Stage 2 fine-tuning on this candidate
            if enable_rotation_testing:
                stage2_result = stage2_fine_tuning_with_rotation(
                    volume_t, slice_t, candidate, search_radius, 
                    use_segmentation, segmentation_threshold,
                    preserve_holes=preserve_holes,
                    w_dice_fine=w_dice_fine, w_ncc_fine=w_ncc_fine, w_ssim_fine=w_ssim_fine, w_orb_fine=w_orb_fine
                )
            else:
                stage2_result = stage2_fine_tuning(
                    volume_t, slice_t, candidate, search_radius, 
                    use_segmentation, segmentation_threshold,
                    preserve_holes=preserve_holes
                )
        
        # Add candidate information to the result
        stage2_result["stage1_score"] = candidate["score"]
        stage2_result["candidate_id"] = candidate["candidate_id"]
        stage2_result["stage1_rank"] = i
        
        stage2_results.append(stage2_result)
        
        if stage2_result["score"] > 0:
            improvement = stage2_result["score"] - candidate["score"]
            # st.write(f"    ✅ Stage 2 score: {stage2_result['score']:.4f} (improvement: {improvement:+.4f})")
        else:
            # st.write(f"    ❌ Stage 2 failed for this candidate")
            pass
    
    # Find the best result after Stage 2
    valid_stage2_results = [r for r in stage2_results if r["score"] > 0]
    
    if not valid_stage2_results:
        st.warning("⚠️ All Stage 2 fine-tuning attempts failed - using best Stage 1 result")
        best_candidate = top_candidates[0]
        return {
            "mode": "multi_candidate_two_stage",
            "stage1_candidates": top_candidates,
            "stage2_results": stage2_results,
            "final_score": best_candidate["score"],
            "final_pose": {"axis": best_candidate["axis"], "index": best_candidate["index"]},
            "score": best_candidate["score"],
            "axis": best_candidate["axis"],
            "index": best_candidate["index"],
            "slice_2d": best_candidate.get("slice_2d"),
            "dice": best_candidate.get("dice", 0),
            "ncc": 0,
            "ssim": 0,
            "orb": 0,  # Add ORB score (0 for Stage 1 fallback)
            "orb_matches": 0,  # Add ORB matches count
            "best_rotation": 0,
            "winning_candidate": 1  # First candidate from Stage 1
        }
    
    # Sort Stage 2 results by final score
    valid_stage2_results.sort(key=lambda x: x["score"], reverse=True)
    final_winner = valid_stage2_results[0]
    
    # Enhanced success message with multi-candidate analysis
    rotation_info = ""
    if enable_rotation_testing and final_winner.get('best_rotation', 0) != 0:
        rotation_info = f" (rotated {final_winner['best_rotation']}°)"
    
    winning_rank = final_winner["stage1_rank"]
    axis_name = ["axial", "coronal", "sagittal"][final_winner["axis"]]
    
    # Comment out verbose winner announcement for cleaner UI
    # st.success(f"🏆 **Multi-Candidate Winner**: Candidate #{winning_rank} ({axis_name} slice {final_winner['index']})")
    # st.success(f"    📈 Final score: {final_winner['score']:.4f}{rotation_info}")
    # st.success(f"    📊 Stage 1→2 improvement: {final_winner['score'] - final_winner['stage1_score']:+.4f}")
    
    # Show ranking changes (commented out to reduce verbose output)
    # st.write("📋 **Candidate Performance Summary:**")
    # for i, result in enumerate(valid_stage2_results, 1):
    #     stage1_rank = result["stage1_rank"]
    #     rank_change = stage1_rank - i  # Positive = moved up, negative = moved down
    #     rank_symbol = "⬆️" if rank_change > 0 else "⬇️" if rank_change < 0 else "➡️"
    #     stage2_rank_text = f"#{i}" if i <= 3 else f"#{i}"
    #     
    #     st.write(f"  {stage2_rank_text} {result['candidate_id']} - Stage 2: {result['score']:.4f} "
    #             f"(was #{stage1_rank} in Stage 1) {rank_symbol}")
    
    # Compile complete result
    complete_result = {
        "mode": "multi_candidate_rotation_invariant" if enable_rotation_testing else "multi_candidate_two_stage",
        "stage1_candidates": top_candidates,
        "stage2_results": stage2_results,
        "final_score": final_winner["score"],
        "final_pose": final_winner["final_pose"],
        # For compatibility with existing display code
        "score": final_winner["score"],
        "axis": final_winner["axis"],
        "index": final_winner["index"],
        "slice_2d": final_winner.get("slice_2d"),
        "dice": final_winner.get("dice", 0),
        "ncc": final_winner.get("ncc", 0),
        "ssim": final_winner.get("ssim", 0),
        "orb": final_winner.get("orb", 0),  # Add ORB score
        "orb_matches": final_winner.get("orb_matches", 0),  # Add ORB matches count
        "best_rotation": final_winner.get("best_rotation", 0),
        "rotation_tested": enable_rotation_testing,
        "winning_candidate": winning_rank,
        "total_candidates_processed": len(top_candidates),
        "candidates_ranking_change": sum(1 for i, r in enumerate(valid_stage2_results) if r["stage1_rank"] != (i + 1))
    }
    
    return complete_result


def two_stage_coarse_to_fine_matching(volume_t, slice_t, axes=[0,1,2], fossil_area_threshold=0.05, 
                                     search_radius=5, use_segmentation=False, segmentation_threshold=0.5,
                                     enable_rotation_testing=True, preserve_holes=True):
    """
    Complete two-stage coarse-to-fine matching pipeline with rotation invariance.
    
    IMPROVED: Now includes rotation testing in Stage 2 for rotation-invariant matching.
    
    Args:
        volume_t: 4D torch tensor (1,1,D,H,W) - the 3D volume
        slice_t: 4D torch tensor (1,1,H,W) - the query slice
        axes: List of axes to search
        fossil_area_threshold: Minimum fossil content for valid slices
        search_radius: Fine-tuning search radius around coarse result
        use_segmentation: Whether to apply advanced segmentation consistently
        segmentation_threshold: Threshold for segmentation (if enabled)
        enable_rotation_testing: Whether to test rotations in Stage 2 (NEW)
    
    Returns:
        final_result: Dict containing complete matching results
    """
    # Stage 1: Coarse Search (now with improved Hu moment weighting)
    if use_segmentation:
        # st.write("🔍 **Stage 1: Coarse Search** (Dice + Hu Moments) with segmentation and improved rotation invariance")
        pass
    else:
        # st.write("🔍 **Stage 1: Coarse Search** (Dice + Hu Moments) with improved rotation invariance")
        pass
    
    initial_pose = stage1_coarse_search(volume_t, slice_t, axes, fossil_area_threshold, 
                                       use_segmentation, segmentation_threshold)
    
    if initial_pose["score"] <= 0:
        st.warning("⚠️ Stage 1 failed - no valid initial pose found")
        return {
            "mode": "two_stage",
            "stage1_result": initial_pose,
            "stage2_result": None,
            "final_score": 0,
            "final_pose": None
        }
    
    st.success(f"✅ Stage 1 Complete: axis={initial_pose['axis']}, index={initial_pose['index']}, score={initial_pose['score']:.4f}")
    
    # Stage 2: Fine-Tuning with optional rotation testing
    if enable_rotation_testing:
        if use_segmentation:
            # st.write("🎯 **Stage 2: Fine-Tuning** (Dice + NCC + SSIM + ORB) with segmentation and rotation testing")
            pass
        else:
            # st.write("🎯 **Stage 2: Fine-Tuning** (Dice + NCC + SSIM + ORB) with rotation testing")
            pass
        
        # Use the new rotation-aware fine-tuning function
        final_result = stage2_fine_tuning_with_rotation(
            volume_t, slice_t, initial_pose, search_radius, 
            use_segmentation, segmentation_threshold,
            preserve_holes=preserve_holes,
            w_dice_fine=w_dice_fine, w_ncc_fine=w_ncc_fine, w_ssim_fine=w_ssim_fine, w_orb_fine=w_orb_fine
        )
    else:
        if use_segmentation:
            # st.write("🎯 **Stage 2: Fine-Tuning** (Dice + NCC + SSIM + ORB) with segmentation")
            pass
        else:
            # st.write("🎯 **Stage 2: Fine-Tuning** (Dice + NCC + SSIM + ORB) without rotation testing")
            pass
        
        # Use the original fine-tuning function
        final_result = stage2_fine_tuning(
            volume_t, slice_t, initial_pose, search_radius, 
            use_segmentation, segmentation_threshold,
            preserve_holes=preserve_holes
        )
    
    if final_result["score"] <= 0:
        st.warning("⚠️ Stage 2 failed - using Stage 1 result")
        # Use Stage 1 result as fallback
        return {
            "mode": "two_stage", 
            "stage1_result": initial_pose,
            "stage2_result": None,
            "final_score": initial_pose["score"],
            "final_pose": {"axis": initial_pose["axis"], "index": initial_pose["index"]},
            "score": initial_pose["score"],
            "axis": initial_pose["axis"],
            "index": initial_pose["index"],
            "slice_2d": initial_pose.get("slice_2d"),
            "dice": initial_pose.get("dice", 0),
            "ncc": 0,
            "ssim": 0,  # Stage 2 failed, so no SSIM computed
            "orb": 0,  # Add ORB score (0 for Stage 1 fallback)
            "orb_matches": 0,  # Add ORB matches count
            "best_rotation": 0
        }
    
    # Enhanced success message with rotation information
    rotation_info = ""
    if enable_rotation_testing and 'best_rotation' in final_result and final_result['best_rotation'] != 0:
        rotation_info = f" (rotated {final_result['best_rotation']}°)"
    
    st.success(f"✅ Stage 2 Complete: axis={final_result['axis']}, index={final_result['index']}, score={final_result['score']:.4f}{rotation_info}")
    
    # Combine results
    complete_result = {
        "mode": "two_stage_rotation_invariant" if enable_rotation_testing else "two_stage",
        "stage1_result": initial_pose,
        "stage2_result": final_result,
        "final_score": final_result["score"],
        "final_pose": final_result["final_pose"],
        # For compatibility with existing display code
        "score": final_result["score"],
        "axis": final_result["axis"],
        "index": final_result["index"],
        "slice_2d": final_result.get("slice_2d"),
        "dice": final_result.get("dice", 0),
        "ncc": final_result.get("ncc", 0),
        "ssim": final_result.get("ssim", 0),  # FIXED: Now using SSIM from Stage 2
        "orb": final_result.get("orb", 0),  # Add ORB score
        "orb_matches": final_result.get("orb_matches", 0),  # Add ORB matches count
        "best_rotation": final_result.get("best_rotation", 0),  # NEW: rotation information
        "rotation_tested": enable_rotation_testing  # NEW: flag indicating if rotation was tested
    }
    
    return complete_result

def extract_arbitrary_slice_torch(volume_t, angles, center, out_shape=(256,256)):
    D, H, W = volume_t.shape[2:]
    ax, ay, az = [a*math.pi/180 for a in angles]
    Rx = torch.tensor([[1,0,0],[0, math.cos(ax), -math.sin(ax)],[0, math.sin(ax), math.cos(ax)]],
                      dtype=torch.float32, device=volume_t.device)
    Ry = torch.tensor([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]],
                      dtype=torch.float32, device=volume_t.device)
    Rz = torch.tensor([[math.cos(az),-math.sin(az),0],[math.sin(az),math.cos(az),0],[0,0,1]],
                      dtype=torch.float32, device=volume_t.device)
    R  = Rz @ Ry @ Rx
    h_out, w_out = out_shape
    xs = torch.linspace(-(w_out-1)/2, (w_out-1)/2, w_out, device=volume_t.device)
    ys = torch.linspace(-(h_out-1)/2, (h_out-1)/2, h_out, device=volume_t.device)
    Yg, Xg = torch.meshgrid(ys, xs, indexing='xy')
    local_pts = torch.stack([Xg.reshape(-1), Yg.reshape(-1), torch.zeros_like(Xg).reshape(-1)])
    rot_pts   = R @ local_pts
    cx, cy, cz = center
    rot_pts[0]+=cx; rot_pts[1]+=cy; rot_pts[2]+=cz
    x_n = 2*rot_pts[0]/(W-1)-1; y_n = 2*rot_pts[1]/(H-1)-1; z_n = 2*rot_pts[2]/(D-1)-1
    grid = torch.stack([z_n, y_n, x_n], -1).view(1,1,h_out,w_out,3)
    return F.grid_sample(volume_t, grid, mode='bilinear',
                         padding_mode='border', align_corners=True).squeeze(2)

def search_orthogonal_torch(volume_t, slice_t, w_ssim, w_ncc, axes=[0,1,2], fossil_area_threshold=0.05, 
                           use_segmentation=False, segmentation_threshold=0.5, preserve_holes=True):
    best = {"score": -1}
    D, H, W = volume_t.shape[2:]
    
    # Validate slice_t dimensions
    if slice_t.dim() < 4:
        st.error(f"❌ Invalid slice tensor dimensions: {slice_t.shape}. Expected 4D tensor (B,C,H,W)")
        return {"score": 0.0, "ssim": 0.0, "ncc": 0.0, "mode": "failed", 
                "axis": None, "index": None, "slice_2d": slice_t,
                "fossil_bbox": None}
    
    h_s, w_s = slice_t.shape[2:]
    
    # Convert volume back to numpy for mask calculation
    vol_np = volume_t.cpu().numpy()[0, 0]
    
    # Build fossil mask using the same method as dataset creation
    thr = smart_threshold(vol_np)
    try:
        mask, bbox = build_fossil_mask(vol_np, thr)
    except (ValueError, IndexError) as e:
        st.warning(f"⚠️ Could not detect fossil content in volume: {e}")
        # Return a basic result if fossil detection fails
        return {"score": 0.0, "ssim": 0.0, "ncc": 0.0, "mode": "failed", 
                "axis": None, "index": None, "slice_2d": slice_t.clone(),
                "fossil_bbox": None}
    
    # Additional safety check for valid bbox
    if len(bbox) != 6:
        st.warning("⚠️ Invalid bounding box detected")
        return {"score": 0.0, "ssim": 0.0, "ncc": 0.0, "mode": "failed", 
                "axis": None, "index": None, "slice_2d": slice_t.clone(),
                "fossil_bbox": None}
    
    # Extract bounding box coordinates
    x0, x1, y0, y1, z0, z1 = bbox
    
    # st.write(f"🔍 Fossil bounding box: X[{x0}:{x1}], Y[{y0}:{y1}], Z[{z0}:{z1}]")
    
    # if use_segmentation:
    #     st.write("🎯 **Enhanced Fossil Segmentation:** Applying identical segmentation to all volume slices for fair comparison")
    
    for axis in axes:
        # Get valid indices that have fossil content within bounding box
        valid_indices = valid_idx(mask, axis, bbox)
        
        if len(valid_indices) == 0:
            continue
            
        # st.write(f"📊 Axis {axis}: Checking {len(valid_indices)} slices within fossil bounding box")
        
        for idx in valid_indices:
            # Check if this slice has enough fossil content
            if axis == 0:
                mask_slice = mask[idx, :, :]
                # Focus on bounding box region only
                mask_slice_cropped = mask_slice[y0:y1+1, z0:z1+1] if y1 < mask.shape[1] and z1 < mask.shape[2] else mask_slice
                if not slice_has_fossil(mask_slice_cropped, min_area_ratio=fossil_area_threshold):
                    continue
                sl = volume_t[:, :, idx, :, :]
            elif axis == 1:
                mask_slice = mask[:, idx, :]
                mask_slice_cropped = mask_slice[x0:x1+1, z0:z1+1] if x1 < mask.shape[0] and z1 < mask.shape[2] else mask_slice
                if not slice_has_fossil(mask_slice_cropped, min_area_ratio=fossil_area_threshold):
                    continue
                sl = volume_t[:, :, :, idx, :].permute(0, 1, 3, 2)
            else:
                mask_slice = mask[:, :, idx]
                mask_slice_cropped = mask_slice[x0:x1+1, y0:y1+1] if x1 < mask.shape[0] and y1 < mask.shape[1] else mask_slice
                if not slice_has_fossil(mask_slice_cropped, min_area_ratio=fossil_area_threshold):
                    continue
                sl = volume_t[:, :, :, :, idx]
            
            if sl.shape[2:] != (h_s, w_s):
                sl = F.interpolate(sl, size=(h_s, w_s), mode='bilinear', align_corners=False)
            
            # Apply segmentation to volume slice if enabled
            if use_segmentation:
                sl = segment_volume_slice(sl, segmentation_threshold, preserve_holes)
            
            ssim_v = ssim_torch(slice_t, sl)
            ncc_v = ncc_torch(slice_t, sl)
            c = combined_similarity_torch(ssim_v, ncc_v, w_ssim, w_ncc)
            
            if c > best.get("score", -1):
                best = dict(score=c, axis=axis, index=idx, slice_2d=sl.clone(),
                           ssim=(ssim_v+1)/2, ncc=(ncc_v+1)/2,
                           fossil_bbox=bbox)
    
    return best

def search_diagonal_brute_force_torch(volume_t, slice_t, w_ssim, w_ncc, angle_step=30, use_all_degrees=False, angle_list=None):
    D,H,W = volume_t.shape[2:]
    center = ((W-1)/2,(H-1)/2,(D-1)/2)
    out_shape = slice_t.shape[2:]
    if angle_list is not None:
        angle_rng = angle_list
    else:
        angle_rng = range(181) if use_all_degrees else range(0,181,angle_step)

    best = {"score":-1}
    for ax in angle_rng:
        for ay in angle_rng:
            for az in angle_rng:
                sl = extract_arbitrary_slice_torch(volume_t,(ax,ay,az),center,out_shape)
                ssim_v = ssim_torch(slice_t,sl)
                ncc_v  = ncc_torch(slice_t,sl)
                c      = combined_similarity_torch(ssim_v,ncc_v,w_ssim,w_ncc)
                if c > best.get("score",-1):
                    best = dict(score=c, angles=(ax,ay,az), slice_2d=sl.clone(),
                                ssim=(ssim_v+1)/2, ncc=(ncc_v+1)/2)
    return best

def search_diagonal_coarse_to_fine_torch(volume_t, slice_t, w_ssim, w_ncc, coarse_step=30, angle_list=None):
    coarse = search_diagonal_brute_force_torch(volume_t, slice_t, w_ssim, w_ncc,
                                               angle_step=coarse_step,
                                               angle_list=angle_list)
    ax_c,ay_c,az_c = coarse["angles"]
    best, best_score = coarse, coarse["score"]
    D,H,W   = volume_t.shape[2:]
    center  = ((W-1)/2,(H-1)/2,(D-1)/2)
    delta, step = 10, 2
    out_shape = slice_t.shape[2:]
    def local(v): 
        return range(max(0,v-delta), min(180,v+delta)+1, step)
    for ax in local(ax_c):
        for ay in local(ay_c):
            for az in local(az_c):
                sl = extract_arbitrary_slice_torch(volume_t,(ax,ay,az),center,out_shape)
                ssim_v = ssim_torch(slice_t,sl)
                ncc_v  = ncc_torch(slice_t,sl)
                c      = combined_similarity_torch(ssim_v,ncc_v,w_ssim,w_ncc)
                if c > best_score:
                    best, best_score = dict(score=c, angles=(ax,ay,az),
                                            slice_2d=sl.clone(),
                                            ssim=(ssim_v+1)/2, ncc=(ncc_v+1)/2), c
    return best

def match_slice_to_volume_torch(volume_t, slice_t, w_ssim=0.5, w_ncc=0.5, axes=[0,1,2],
                                enable_diagonal=False, angle_step=30, use_all_degrees=False,
                                coarse_to_fine=False, angle_list=None, fossil_area_threshold=0.05,
                                use_segmentation=False, segmentation_threshold=0.5, 
                                use_two_stage=False, enable_rotation_testing=True, preserve_holes=True,
                                use_multi_candidate=False, top_n_candidates=5,
                                w_dice_fine=0.3, w_ncc_fine=0.25, w_ssim_fine=0.25, w_orb_fine=0.2):
    """
    Enhanced matching function with multiple pipeline options.
    
    Args:
        use_two_stage: If True, uses the two-stage pipeline (Dice+Hu -> Dice+NCC)
                      If False, uses the original single-stage pipeline (SSIM+NCC)
        use_multi_candidate: If True, uses the NEW multi-candidate approach
        enable_rotation_testing: Always True (rotation testing always enabled)
        top_n_candidates: Number of candidates to process in multi-candidate mode
    """
    
    # Only Multi-Candidate Two-Stage Pipeline with Rotation Invariance is available
    # Other pipelines (two-stage and single-stage) have been removed
    
    return multi_candidate_fine_tuning(
        volume_t, slice_t, axes, fossil_area_threshold, search_radius=5,
        use_segmentation=use_segmentation, segmentation_threshold=segmentation_threshold,
        enable_rotation_testing=enable_rotation_testing, preserve_holes=preserve_holes,
        top_n_candidates=top_n_candidates,
        w_dice_fine=w_dice_fine, w_ncc_fine=w_ncc_fine, w_ssim_fine=w_ssim_fine, w_orb_fine=w_orb_fine,
        enable_diagonal=enable_diagonal, angle_list=angle_list
    )

def get_orientation_name(axis):
    """Convert axis number to orientation name"""
    if axis == 0:
        return "axial"
    elif axis == 1:
        return "coronal"
    elif axis == 2:
        return "sagittal"
    else:
        return "diagonal"

def orthogonal_slice_plane_coords(shape, slice_idx, orientation, fx=1, fy=1, fz=1):
    D, H, W = shape
    
    # Map orientation name to axis number
    if orientation == "axial":
        axis = 0
    elif orientation == "coronal":
        axis = 1
    elif orientation == "sagittal":
        axis = 2
    else:
        axis = 0  # default to axial
    
    if axis == 0:  # Z-axis slice (XY plane at depth slice_idx) - AXIAL
        # Create coordinates that match the volume's coordinate system
        y = np.arange(H)  # Note: using H for Y dimension
        x = np.arange(W)  # Note: using W for X dimension
        X, Y = np.meshgrid(x, y, indexing='xy')
        Z = np.full_like(X, slice_idx, dtype=float)
        return X, Y, Z
    elif axis == 1:  # Y-axis slice (XZ plane at height slice_idx) - CORONAL
        x = np.arange(W)
        z = np.arange(D)
        X, Z = np.meshgrid(x, z, indexing='xy')
        Y = np.full_like(X, slice_idx, dtype=float)
        return X, Y, Z
    else:  # X-axis slice (YZ plane at width slice_idx) - SAGITTAL
        y = np.arange(H)
        z = np.arange(D)
        Y, Z = np.meshgrid(y, z, indexing='xy')
        X = np.full_like(Y, slice_idx, dtype=float)
        return X, Y, Z

def diagonal_slice_plane_coords(shape, angles, out_shape):
    D, H, W = shape
    ax, ay, az = [np.deg2rad(a) for a in angles]
    
    # Create rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)], 
                   [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx
    
    # Volume center
    cx, cy, cz = (W - 1) / 2, (H - 1) / 2, (D - 1) / 2
    
    # Create local coordinate system for the slice
    h_out, w_out = out_shape
    xs = np.linspace(-(w_out - 1) / 2, (w_out - 1) / 2, w_out)
    ys = np.linspace(-(h_out - 1) / 2, (h_out - 1) / 2, h_out)
    Yg, Xg = np.meshgrid(ys, xs, indexing='xy')
    
    # Local points in the slice plane (Z=0 in local coordinates)
    local_points = np.stack([Xg.ravel(), Yg.ravel(), np.zeros_like(Xg.ravel())])
    
    # Transform to world coordinates
    world_points = R @ local_points
    
    # Translate to volume center
    world_points[0] += cx
    world_points[1] += cy  
    world_points[2] += cz
    
    # Reshape back to 2D grid
    X_world = world_points[0].reshape(h_out, w_out)
    Y_world = world_points[1].reshape(h_out, w_out)
    Z_world = world_points[2].reshape(h_out, w_out)
    
    return X_world, Y_world, Z_world

# =================================================================
# Volume processing helper functions
# =================================================================
def smart_threshold(volume, percentile=85):
    """Smart thresholding for fossil detection"""
    non_zero_values = volume[volume > 0]
    if len(non_zero_values) == 0:
        return 0
    return np.percentile(non_zero_values, percentile)

def valid_idx(mask, axis, bbox):
    """Get valid slice indices that contain fossil content within bounding box"""
    x0, x1, y0, y1, z0, z1 = bbox
    valid_indices = []
    
    if axis == 0:  # Z-axis slices
        for idx in range(z0, min(z1 + 1, mask.shape[0])):
            slice_mask = mask[idx, y0:y1+1, :]
            if np.sum(slice_mask) > 0:
                valid_indices.append(idx)
    elif axis == 1:  # Y-axis slices  
        for idx in range(y0, min(y1 + 1, mask.shape[1])):
            slice_mask = mask[x0:x1+1, idx, z0:z1+1]
            if np.sum(slice_mask) > 0:
                valid_indices.append(idx)
    else:  # X-axis slices
        for idx in range(x0, min(x1 + 1, mask.shape[2])):
            slice_mask = mask[:, y0:y1+1, idx]
            if np.sum(slice_mask) > 0:
                valid_indices.append(idx)
    
    return valid_indices

def slice_has_fossil(mask_slice, min_area_ratio=0.05):
    """Check if a slice has enough fossil content"""
    fossil_pixels = np.sum(mask_slice > 0)
    total_pixels = mask_slice.size
    return (fossil_pixels / total_pixels) >= min_area_ratio

def segment_volume_slice(slice_tensor, threshold=0.5, preserve_holes=True):
    """Apply segmentation to a volume slice for fair comparison"""
    # Convert tensor to numpy for segmentation
    if hasattr(slice_tensor, 'cpu'):
        slice_np = slice_tensor.cpu().numpy().squeeze()
        device = slice_tensor.device
    else:
        slice_np = slice_tensor.squeeze()
        device = None
    
    # Apply the full fossil segmentation (same as input image)
    try:
        segmented_np = apply_fossil_segmentation(slice_np, threshold, show_preview=False, preserve_holes=preserve_holes)
        
        # Convert back to tensor
        if device is not None:
            segmented_tensor = torch.from_numpy(segmented_np).float().to(device)
        else:
            segmented_tensor = torch.from_numpy(segmented_np).float()
        
        # Ensure proper dimensions
        if segmented_tensor.dim() == 2:
            segmented_tensor = segmented_tensor.unsqueeze(0).unsqueeze(0)
        elif segmented_tensor.dim() == 3:
            segmented_tensor = segmented_tensor.unsqueeze(0)
            
        return segmented_tensor
    except:
        # Fallback to simple thresholding if full segmentation fails
        slice_norm = (slice_tensor - slice_tensor.min()) / (slice_tensor.max() - slice_tensor.min() + 1e-8)
        segmented = (slice_norm > threshold).float()
        return segmented

# [Keep all SSIM, NCC, and matching functions from original]
def create_gaussian_window(window_size=11, sigma=1.5, channels=1):
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= (window_size - 1) / 2.0
    gauss  = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    gauss2d = (gauss.unsqueeze(0) * gauss.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
    return gauss2d.expand(channels, 1, window_size, window_size)

def ncc_torch(img1, img2):
    f1, f2   = img1.view(-1), img2.view(-1)
    mean1    = torch.mean(f1); mean2 = torch.mean(f2)
    num      = torch.sum((f1-mean1)*(f2-mean2))
    denom    = torch.sqrt(torch.sum((f1-mean1)**2)*torch.sum((f2-mean2)**2) + 1e-8)
    return (num/denom).item()

def combined_similarity_torch(ssim_val, ncc_val, w_ssim=0.5, w_ncc=0.5):
    ssim_norm = (ssim_val+1)/2; ncc_norm = (ncc_val+1)/2
    return w_ssim*ssim_norm + w_ncc*ncc_norm

# Initialize session state
if 'matching_results' not in st.session_state:
    st.session_state.matching_results = []
if 'best_match_info' not in st.session_state:
    st.session_state.best_match_info = None
if 'matching_done' not in st.session_state:
    st.session_state.matching_done = False
if 'loaded_volumes' not in st.session_state:
    st.session_state.loaded_volumes = {}

# Enhanced sidebar with modern design
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;">
            <h2 style="margin: 0; color: #ffffff; font-size: 1.5rem;">🎛️ Control Center</h2>
            <p style="margin: 0.5rem 0 0 0; color: #c7d2fe; font-size: 0.9rem;">Configure analysis parameters</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced reset button with compact centered styling
    if st.button("🔄 Reset Application", type="primary", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("""
        <div style="height: 1px; background: linear-gradient(90deg, transparent, #374151, transparent); margin: 1rem 0;"></div>
    """, unsafe_allow_html=True)
    
    # Enhanced search configuration section
    st.markdown("""
        <div class="glass-card" style="padding: 0.8rem 1rem; margin-bottom: 0.5rem; text-align: center;">
            <h4 style="margin: 0; color: #ffffff; display: flex; align-items: center; justify-content: center;">
                <span class="icon">🔍</span>Search Configuration
            </h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Add spacing between Search Configuration and Pipeline Selection
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    # NEW: Pipeline Selection
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h5 style="margin: 0 0 0.5rem 0; color: #c7d2fe; font-weight: 500;">
                <span class="icon">⚙️</span>Matching Pipeline
            </h5>
        </div>
    """, unsafe_allow_html=True)
    
    # Only Multi-Candidate Fine-Tuning pipeline is available
    pipeline_mode = "Multi-Candidate Fine-Tuning"
    
    # Always use multi-candidate approach - other pipelines removed
    use_two_stage = False
    use_multi_candidate = True
    
    # Multi-Candidate Pipeline is now the only option
    st.markdown("""
        <div style="padding: 0.75rem; background: rgba(139, 69, 19, 0.1); border: 1px solid rgba(139, 69, 19, 0.3); border-radius: 8px; margin-top: 0.5rem;">
            <h5 style="margin: 0 0 0.5rem 0; color: #d2691e;">🏆 Multi-Candidate Pipeline</h5>
            <small style="color: #daa520;">
                <strong>Strategy:</strong> Processes multiple promising candidates<br>
                <strong>Stage 1:</strong> Finds top N candidates (Dice + Hu Moments)<br>
                <strong>Stage 2:</strong> Fine-tunes each candidate (Dice + NCC + SSIM + ORB)<br>
                <strong>Winner:</strong> Best performer after Stage 2 processing<br>
                <strong>Benefit:</strong> Prevents early dismissal of good matches
            </small>
        </div>
    """, unsafe_allow_html=True)
    
    # Rotation invariance option for multi-candidate pipeline (always enabled)
    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
    
    # Always enable rotation testing - no user option needed
    enable_rotation_testing = True
    
    # # Always show the rotation testing info since it's always enabled
    # st.markdown(f"""
    #     <div style="padding: 0.75rem; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; margin-top: 0.5rem;">
    #         <h5 style="margin: 0 0 0.5rem 0; color: #f59e0b;">🎯 Multi-Candidate + Rotation Testing (Always Active)</h5>
    #         <small style="color: #fbbf24;">
    #             <strong>Power Mode:</strong> Tests {top_n_candidates} candidates × 12 rotations<br>
    #             <strong>Total Tests:</strong> Up to {top_n_candidates * 12} fine-tuning operations<br>
    #             <strong>Quality:</strong> Maximum possible accuracy<br>
    #             <strong>Time:</strong> Most thorough but slower processing
    #         </small>
    #     </div>
    # """, unsafe_allow_html=True)
    
    # # Add comparison information with new pipeline
    # with st.expander("📊 Pipeline Comparison", expanded=False):
    #     st.markdown("### 🆚 Pipeline Comparison Overview")
        
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         st.markdown("#### 🔧 Single-Stage")
    #         st.write("**Metrics:** SSIM + NCC")
    #         st.write("**Search:** Direct similarity")
    #         st.write("**Speed:** Fastest")
    #         st.write("**Best for:** Clean fossils")
        
    #     with col2:
    #         st.markdown("#### 🚀 Two-Stage")
    #         st.write("**Stage 1:** Dice + Hu Moments")
    #         st.write("**Stage 2:** Dice + NCC + SSIM")
    #         st.write("**Speed:** Moderate")
    #         st.write("**Best for:** Complex fossils")
        
    #     with col3:
    #         st.markdown("#### 🏆 Multi-Candidate")
    #         st.write("**Innovation:** Process top N candidates")
    #         st.write("**Method:** Best of multiple fine-tunings")
    #         st.write("**Speed:** Thorough but slower")
    #         st.write("**Best for:** Maximum accuracy")
        
    #     st.markdown("---")
    #     st.markdown("### 🎯 When to Use Each Pipeline")
        
    #     st.markdown("**🔧 Single-Stage:** Use when you have high-quality, clean fossil images and need fast results.")
    #     st.markdown("**🚀 Two-Stage:** Use for most fossil identification tasks. Good balance of accuracy and speed.")
    #     st.markdown("**🏆 Multi-Candidate:** Use when accuracy is paramount and you want to ensure no good matches are missed.")
        
    #     st.info("💡 **Recommendation:** Start with Multi-Candidate for best results, fall back to Two-Stage if processing time is a concern.")
    
    
    # Add spacing between Search Configuration and Orthogonal Search Axes
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    # Enhanced orthogonal axes selection
    st.markdown("""
        <div style="margin-bottom: 1rem;">
            <h5 style="margin: 0 0 0.5rem 0; color: #c7d2fe; font-weight: 500;">
                <span class="icon">⚙️</span>Orthogonal Search Axes
            </h5>
        </div>
    """, unsafe_allow_html=True)
    
    axes_options = st.multiselect(
        "",
        [0, 1, 2],
        default=[0, 1, 2],
        format_func=lambda x: ["🔵 Axial (Z-axis)", "🟢 Coronal (Y-axis)", "🟡 Sagittal (X-axis)"][x],
        help="Select which anatomical planes to search through",
        label_visibility="collapsed"
    )
    
    # Visual indicator for selected axes
    if axes_options:
        selected_names = [["Axial", "Coronal", "Sagittal"][i] for i in axes_options]
        st.markdown(f"""
            <div style="padding: 0.75rem; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; margin-top: 0.5rem;">
                <small style="color: #10b981;">✓ Active: {', '.join(selected_names)}</small>
            </div>
        """, unsafe_allow_html=True)
    
    # Set default fossil area threshold
    fossil_area_threshold = 0.05
    
    st.markdown("""
        <div style="margin: 1.5rem 0; padding: 1rem; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px;">
            <small style="color: #60a5fa;">
                💡 <strong>Note:</strong> Image segmentation is controlled in the Upload & Process tab. 
                This setting controls fossil region detection in 3D volumes.
            </small>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced diagonal search section
    st.markdown("""
        <div class="glass-card" style="padding: 0.8rem 1rem; margin-bottom: 0.5rem; text-align: center;">
            <h4 style="margin: 0; color: #ffffff; display: flex; align-items: center; justify-content: center;">
                <span class="icon">↗️</span>Advanced Diagonal Search
            </h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Add spacing between Advanced Diagonal Search and Enable Diagonal Search
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    enable_diag = st.checkbox("🚀 Enable Diagonal Search", value=False, 
                              help="Activate advanced 3D angular search for complex orientations")
    
    if enable_diag:
        st.markdown("""
            <div style="margin: 1rem 0; padding: 1rem; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px;">
                <h5 style="margin: 0 0 0.5rem 0; color: #f59e0b;">⚡ Advanced Search Mode</h5>
            </div>
        """, unsafe_allow_html=True)
        
        angle_mode = st.radio(
            "Angular Search Strategy:",
            ["🎯 Preset Optimal", "⚙️ Custom Step", "🌐 Full Spectrum"],
            format_func=lambda x: x.split(" ", 1)[1],
            help="Choose the angular search approach for diagonal matching"
        )
        
        if "Custom Step" in angle_mode:
            angle_step = st.slider("🎚️ Angular Step Size", 5, 90, 30, 5, 
                                  help="Smaller steps = higher precision but slower processing")
            st.markdown(f"""
                <div style="padding: 0.5rem; background: rgba(99, 102, 241, 0.1); border-radius: 6px; margin-top: 0.5rem;">
                    <small style="color: #6366f1;">Estimated angles to test: ~{(180//angle_step)**3:,}</small>
                </div>
            """, unsafe_allow_html=True)
        elif "Full Spectrum" in angle_mode:
            use_all_deg = True
            st.markdown("""
                <div style="padding: 1rem; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; margin-top: 0.5rem;">
                    <h5 style="margin: 0 0 0.5rem 0; color: #ef4444;">⚠️ Performance Warning</h5>
                    <small style="color: #fca5a5;">Full spectrum search will test 180³ = 5.8M combinations. This may take hours!</small>
                </div>
            """, unsafe_allow_html=True)
        else:  # Preset
            use_angle_list = True
            st.markdown(f"""
                <div style="padding: 0.75rem; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; margin-top: 0.5rem;">
                    <h5 style="margin: 0 0 0.5rem 0; color: #10b981;">✨ Optimized Search</h5>
                    <small style="color: #6ee7b7;">Using scientifically optimized angles: {ROTATION_ANGLES}</small>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="padding: 0.75rem; background: rgba(107, 114, 128, 0.1); border: 1px solid rgba(107, 114, 128, 0.3); border-radius: 8px; margin-top: 0.5rem;">
                <small style="color: #9ca3af;">🔒 Diagonal search disabled - using orthogonal planes only</small>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="height: 1px; background: linear-gradient(90deg, transparent, #374151, transparent); margin: 2rem 0;"></div>
    """, unsafe_allow_html=True)

# Enhanced main content with sophisticated tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "**📤 Upload & Setup**", 
    "**🗂️ Model Selection**", 
    "**🎯 Run Analysis**",
    "**📊 Results & Stats**",
    "**🔬 3D Visualization**"
])

with tab1:
    st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h2 style="margin: 0 0 0.5rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center; text-align: center;">
                <span class="icon">📤</span>Fossil Image Upload & Processing
            </h2>
            <p style="color: #c7d2fe; font-size: 1.1rem; margin: 0; text-align: center;">
                Upload your fossil slice image and configure intelligent processing parameters
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced file uploader section
    st.markdown("""
        <div class="glass-card" style="padding: 2rem; margin-bottom: 2rem;">
            <h3 style="margin: 0 0 1rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center; text-align: center;">
                <span class="icon">🖼️</span>Image Upload Center
            </h3>
            <p style="color: #c7d2fe; margin-bottom: 1.5rem; text-align: center;">
                Select a high-resolution fossil slice image for AI-powered identification
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Enhanced file uploader with better styling
    uploaded_slice = st.file_uploader(
        "",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        help="Supported formats: PNG, JPG, JPEG, TIFF, BMP | Max size: 200MB",
        label_visibility="collapsed"
    )
    
    if uploaded_slice:
        # Auto-restart logic with enhanced feedback
        current_file_hash = hash(uploaded_slice.getvalue())
        
        if 'previous_file_hash' not in st.session_state:
            st.session_state.previous_file_hash = current_file_hash
            st.markdown("""
                <div class="success-card">
                    <h4 style="margin: 0 0 0.5rem 0;">🎉 Image Successfully Uploaded!</h4>
                    <p style="margin: 0;">Ready for intelligent processing and species identification</p>
                </div>
            """, unsafe_allow_html=True)
        elif st.session_state.previous_file_hash != current_file_hash:
            st.session_state.previous_file_hash = current_file_hash
            st.markdown("""
                <div class="info-card">
                    <h4 style="margin: 0 0 0.5rem 0;">🔄 New Image Detected</h4>
                    <p style="margin: 0;">Previous analysis cleared - processing fresh upload</p>
                </div>
            """, unsafe_allow_html=True)
            # Clear relevant session state
            st.session_state.matching_results = []
            st.session_state.best_match_info = None
            st.session_state.matching_done = False
        
        # Process uploaded image
        pil_orig = Image.open(uploaded_slice).convert("RGB")
        pil_orig_for_display = Image.open(uploaded_slice).convert("RGB")
        # Enhanced white background detection
        raw_img_np = np.array(pil_orig)
        raw_img_gray = cv2.cvtColor(raw_img_np, cv2.COLOR_RGB2GRAY)
        
        # Check if the average brightness is high (indicative of a white background)
        if np.mean(raw_img_gray) > 190:
            st.markdown("""
                <div style="padding: 1rem; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; margin: 1rem 0;">
                    <h5 style="margin: 0 0 0.5rem 0; color: #60a5fa;">� Smart Color Inversion</h5>
                    <p style="margin: 0; color: #93c5fd;">White background detected - automatically inverting image colors for optimal AI processing</p>
                </div>
            """, unsafe_allow_html=True)
            # Invert the original PIL image in-place for all subsequent steps
            pil_orig = Image.fromarray(255 - raw_img_np)
        
        width, height = pil_orig.size
        file_size = len(uploaded_slice.getvalue())
        
        # Enhanced image display with sophisticated metrics
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display the potentially inverted image with better styling
            st.image(pil_orig_for_display, 
                    caption="📸 Uploaded Fossil Image (auto-processed)", 
                    use_container_width=True)
            
        with col2:
            # Sophisticated metrics display
            st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0 0 1rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center; text-align: center; font-size: 1rem;">
                        <span class="icon">📊</span>Image Analytics
                    </h4>
                    <div style="display: grid; gap: 0.75rem;">
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <span style="color: #c7d2fe;">Dimensions:</span>
                            <strong style="color: #ffffff;">{width} × {height}</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <span style="color: #c7d2fe;">Resolution:</span>
                            <strong style="color: #ffffff;">px</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <span style="color: #c7d2fe;">File Size:</span>
                            <strong style="color: #ffffff;">{file_size/1024:.1f} KB</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                            <span style="color: #c7d2fe;">Aspect Ratio:</span>
                            <strong style="color: #ffffff;">{width/height:.2f}:1</strong>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0;">
                            <span style="color: #c7d2fe;">Total Pixels:</span>
                            <strong style="color: #ffffff;">{width*height:,}</strong>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📊 Image Details", expanded=False):
            st.info(f"**📐 Dimensions:** {width} × {height} pixels\n"
                      f"**📁 File Size:** {file_size/1024:.1f} KB ({file_size:,} bytes)\n"
                      f"**🎨 Format:** RGB Color\n"
                      f"**📏 Aspect Ratio:** {width/height:.2f}:1\n"
                      f"**🔢 Total Pixels:** {width*height:,}")
        
        # Enhanced image processing options
        st.markdown("""
            <div style="margin: 2rem 0;">
                <h3 style="margin: 0 0 1rem 0; color: #ffffff; display: flex; align-items: center;">
                    <span class="icon">🛠️</span>Intelligent Image Processing
                </h3>
                <p style="color: #c7d2fe; margin-bottom: 1.5rem;">
                    Choose the optimal preprocessing method for your fossil image
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            image_processing_mode = st.radio(
                "Choose how to process the uploaded image:",
                ["Use Original", "Auto-Detect & Crop", "Manual Crop"],
                index=0,
                help="Select the image processing method before matching"
            )
        
        with col2:
            # Enhanced processing guide
            with st.expander("🧠 AI Processing Guide", expanded=False):
                st.markdown("### 🎯 Auto-Detect & Crop")
                st.write("**Best for:** Images with clear fossil + background")
                st.write("**Technology:** AI-powered boundary detection")
                st.write("**Result:** Automatically finds fossil boundaries")
                
                st.markdown("---")
                
                st.markdown("### ✂️ Manual Crop")
                st.write("**Best for:** Complex images or when auto-detection fails")
                st.write("**Technology:** Interactive selection tool")
                st.write("**Result:** Precise user-controlled cropping")
                
                st.markdown("---")
                
                st.markdown("### 📷 Use Original")
                st.write("**Best for:** Pre-cropped or clean images")
                st.write("**Technology:** Direct processing")
                st.write("**Result:** Minimal background interference")
        
        # Mode-specific controls
        crop_padding = 10
        show_detection_debug = False
        smart_background = True
        
        if image_processing_mode == "Auto-Detect & Crop":
            col1, col2, col3 = st.columns(3)
            with col1:
                crop_padding = st.slider("Crop Padding (pixels)", 0, 50, 10,
                                        help="Extra padding around detected fossil region")
            with col2:
                show_detection_debug = st.checkbox("Show Detection Debug", value=False,
                                                 help="Show intermediate steps of fossil detection")
            with col3:
                smart_background = st.checkbox("Smart Background Fill", value=True,
                                             help="Fill background with intelligent values instead of black")
            st.info("🎯 Automatically finds and crops to fossil content")
            
        elif image_processing_mode == "Manual Crop":
            smart_background = st.checkbox("Smart Background Fill", value=True,
                                         help="Fill background with intelligent values instead of black")
            st.info("✂️ Use the interactive tool below to select a region to crop")
        
        # Store crop coordinates but don't crop yet - we'll crop after resizing
        crop_coords = None
        auto_cropped = False
        manual_cropped = False
        processing_method_used = "original"
        
        if image_processing_mode == "Auto-Detect & Crop":
            # Automatic fossil detection - find coordinates but don't crop yet
            slice_for_detection = np.array(pil_orig).astype(np.float32)
            if slice_for_detection.ndim == 3:
                slice_for_detection = rgb2gray(slice_for_detection)
            
            # Show debug information if requested
            if show_detection_debug:
                st.write("### 🔍 Fossil Detection Debug")
            
            # Detect fossil content
            x_min, x_max, y_min, y_max, fossil_detected = detect_fossil_in_2d_image(slice_for_detection, crop_padding)
            
            if fossil_detected:
                # Store crop coordinates for later use (after resizing)
                crop_coords = (x_min, y_min, x_max + 1, y_max + 1)
                auto_cropped = True
                processing_method_used = "auto_crop"
                
                fossil_area = (x_max - x_min + 1) * (y_max - y_min + 1)
                total_area = slice_for_detection.shape[0] * slice_for_detection.shape[1]
                # st.success(f"✅ **Fossil detected!** (Will be cropped after resizing to preserve scale)\n"
                #           f"- Fossil region: {x_max-x_min+1}×{y_max-y_min+1} pixels\n"
                #           f"- Fossil area: {fossil_area/total_area*100:.1f}% of original image\n"
                #           f"- Crop coordinates: ({x_min}, {y_min}) to ({x_max}, {y_max})")
            else:
                st.warning("⚠️ **No clear fossil content detected**\n"
                          "- Using original image\n"
                          "- Consider switching to 'Manual Crop' mode\n"
                          "- Auto-crop works best with clear fossil boundaries")
                processing_method_used = "original"
        
        elif image_processing_mode == "Manual Crop":
            # Manual cropping option
            if image_selector is not None:
                sel = image_selector(pil_orig, selection_type="box", width=600, height=600)
                
                if sel and sel["selection"]["box"]:
                    show_selection(pil_orig, sel)
                    bx = sel["selection"]["box"][0]
                    
                    # Store crop coordinates for later use (after resizing)
                    crop_coords = (int(bx["x"][0]), int(bx["y"][0]), int(bx["x"][1]), int(bx["y"][1]))
                    manual_cropped = True
                    processing_method_used = "manual_crop"
                    
                    crop_width = int(bx["x"][1]) - int(bx["x"][0])
                    crop_height = int(bx["y"][1]) - int(bx["y"][0])
                    st.success(f"✅ **Manual selection made!** (Will be cropped after resizing to preserve scale)\n"
                               f"- Selected region: {crop_width}×{crop_height} pixels\n"
                               f"- Crop coordinates: ({int(bx['x'][0])}, {int(bx['y'][0])}) to ({int(bx['x'][1])}, {int(bx['y'][1])})")
                else:
                    st.info("👆 **Please draw a box** on the image above to apply a manual crop.")
            else:
                st.error("❌ **Manual cropping not available**\n"
                         "Install streamlit-extras for manual image cropping: `pip install streamlit-extras`")
        
        else:  # "Use Original"
            # st.info("📷 **Using original image** without any modifications")
            processing_method_used = "original"
        
        # Convert original image to numpy and grayscale
        slice_np = np.array(pil_orig).astype(np.float32)
        slice_np = rgb2gray(slice_np) if slice_np.ndim==3 else slice_np
        
        # Store the original PIL image for display purposes (keep as RGB/color)
        # original_pil_for_display = pil_orig.copy()
        original_pil_for_display = pil_orig_for_display.copy()
        # Resize the PIL image for display
        original_pil_resized_for_display = original_pil_for_display.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Resize to 224x224 FIRST (preserves relative fossil scale)
        from skimage.transform import resize
        original_shape = slice_np.shape
        slice_np_resized = resize(slice_np, (224, 224), mode="constant", anti_aliasing=True)
        
        # Apply cropping AFTER resizing (if crop coordinates were detected/selected)
        if crop_coords is not None and processing_method_used in ["auto_crop", "manual_crop"]:
            # Scale the crop coordinates to the resized 224x224 space
            orig_height, orig_width = original_shape
            scale_x = 224 / orig_width
            scale_y = 224 / orig_height
            
            x_min, y_min, x_max, y_max = crop_coords
            
            # Scale coordinates to 224x224 space
            x_min_scaled = int(x_min * scale_x)
            y_min_scaled = int(y_min * scale_y)
            x_max_scaled = int(x_max * scale_x)
            y_max_scaled = int(y_max * scale_y)
            
            # Ensure coordinates are within bounds
            x_min_scaled = max(0, min(x_min_scaled, 223))
            y_min_scaled = max(0, min(y_min_scaled, 223))
            x_max_scaled = max(x_min_scaled + 1, min(x_max_scaled, 224))
            y_max_scaled = max(y_min_scaled + 1, min(y_max_scaled, 224))
            
            # Crop from the resized image
            slice_np_cropped = slice_np_resized[y_min_scaled:y_max_scaled, x_min_scaled:x_max_scaled]
            
            # Smart background filling instead of black padding
            crop_height, crop_width = slice_np_cropped.shape
            
            # Center the cropped fossil in 224x224 space
            pad_y = (224 - crop_height) // 2
            pad_x = (224 - crop_width) // 2
            
            # Choose background fill strategy
            if smart_background:
                # Intelligent background filling for better matching
                # Option 1: Use the border pixels of the cropped region
                border_pixels = np.concatenate([
                    slice_np_cropped[0, :],  # top edge
                    slice_np_cropped[-1, :], # bottom edge
                    slice_np_cropped[:, 0],  # left edge
                    slice_np_cropped[:, -1]  # right edge
                ])
                border_background = np.median(border_pixels)
                
                # Option 2: Use statistical background from the resized image
                background_threshold = np.percentile(slice_np_resized, 25)  # bottom 25% are likely background
                background_mask = slice_np_resized <= background_threshold
                if np.sum(background_mask) > 0:
                    statistical_background = np.mean(slice_np_resized[background_mask])
                else:
                    statistical_background = border_background
                
                # Use the more conservative (lower) value to avoid bright artifacts
                background_value = min(border_background, statistical_background)
            else:
                # Traditional black background
                background_value = 0.0
            
            # Create final image with chosen background
            slice_np_final = np.full((224, 224), background_value, dtype=np.float32)
            slice_np_final[pad_y:pad_y + crop_height, pad_x:pad_x + crop_width] = slice_np_cropped
            
            # st.success(f"🎯 **Fossil detected - Smart-Crop Processing Complete!**\n"
            #           f"- Original size: {orig_width}×{orig_height} pixels\n"
            #           f"- Resized to: 224×224 pixels (fossil scale preserved)\n"
            #           f"- Cropped region in 224×224 space: {crop_width}×{crop_height} pixels\n"
            #           f"- Background: Intelligent filling (value: {background_value:.3f}) instead of black\n"
            #           f"- Maintains context for accurate 3D matching!")
            
            slice_np = slice_np_final
            
        else:
            # No cropping - use the resized image directly
            slice_np = slice_np_resized
            st.info(f"📏 **Standard Processing:** Original {original_shape[1]}×{original_shape[0]} → 224×224 pixels")
        
        # Show image processing results for crop modes
        if processing_method_used in ["auto_crop", "manual_crop"] and crop_coords is not None:
            # Display the resized and cropped images side by side
            col1, col2 = st.columns(2)
            with col1:
                # Normalize for display
                resized_display = (slice_np_resized - slice_np_resized.min()) / (slice_np_resized.max() - slice_np_resized.min() + 1e-8)
                st.image(resized_display, caption="📏 Resized to 224×224 (Scale Preserved)", use_container_width=True)
            with col2:
                # Normalize for display
                final_display = (slice_np - slice_np.min()) / (slice_np.max() - slice_np.min() + 1e-8)
                st.image(final_display, caption="✂️ Cropped in 224×224 Space", use_container_width=True)
        
        # Enhanced segmentation section
        st.markdown("""
            <div style="margin: 2rem 0;">
                <h3 style="margin: 0 0 1rem 0; color: #ffffff; display: flex; align-items: center;">
                    <span class="icon">🎯</span>Advanced Fossil Segmentation
                </h3>
                <p style="color: #c7d2fe; margin-bottom: 1.5rem;">
                    Enable AI-powered fossil extraction for pure structure-to-structure matching
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🧠 Multi-Candidate Pipeline Technology", expanded=False):
            st.markdown("### 🏆 Multi-Candidate Fine-Tuning Algorithm")
            
            st.markdown("#### 🚀 Revolutionary Approach: Process Multiple Promising Candidates")
            st.success("**Key Innovation:** Instead of settling on the first 'best' match from Stage 1, we process multiple promising candidates through Stage 2.")
            
            st.markdown("#### 📋 Conceptual Change vs Traditional Two-Stage:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔧 Traditional Two-Stage:**")
                st.info("1. Stage 1 finds 1 best candidate\n2. Stage 2 fine-tunes that candidate\n3. Result is returned")
                st.warning("**Problem:** Good candidates might be dismissed if Stage 1 scoring isn't perfect")
                
            with col2:
                st.markdown("**🏆 Multi-Candidate Approach:**")
                st.success("1. Stage 1 finds TOP N candidates (e.g., top 5)\n2. Stage 2 fine-tunes ALL candidates\n3. Winner = highest Stage 2 score")
                st.success("**Benefit:** Ensures promising matches aren't lost due to Stage 1 limitations")
            
            st.markdown("#### ⚙️ Implementation Details")
            
            st.markdown("##### 🔍 Stage 1: Multi-Candidate Coarse Search")
            st.write("**Goal:** Cast a wide net to find ALL potentially good matches")
            st.write("- **Process:** Search ALL valid slices across ALL axes")
            st.write("- **Scoring:** Dice Score + Hu Moments for each slice")
            st.write("- **Selection:** Sort by score and keep top N candidates")
            st.write("- **Output:** List of N candidates with their initial scores")
            
            st.markdown("##### 🎯 Stage 2: Multi-Candidate Fine-Tuning")
            st.write("**Goal:** Apply sophisticated analysis to each candidate")
            st.write("- **Process:** Run full fine-tuning on each of the N candidates")
            st.write("- **Metrics:** Dice + NCC + SSIM for detailed evaluation")
            st.write("- **Rotation:** Optional rotation testing for each candidate")
            st.write("- **Winner Selection:** Candidate with highest Stage 2 score wins")
            
            st.markdown("#### 🎯 Key Advantages")
            st.success("✅ **Prevents Early Dismissal:** Good matches aren't lost in Stage 1")
            st.success("✅ **Ranking Improvements:** Lower-ranked Stage 1 candidates can win in Stage 2")
            st.success("✅ **Higher Accuracy:** More thorough evaluation of all possibilities")
            st.success("✅ **Rotation Invariant:** Each candidate tested with multiple rotations")
            st.success("✅ **Robust Results:** Less sensitive to Stage 1 scoring limitations")
            
            st.markdown("#### 📊 Performance Analysis")
            st.write("**Computational Cost:** N times more processing than traditional two-stage")
            st.write("**Quality Benefit:** Significantly improved match accuracy")
            st.write("**Best Use Case:** When accuracy is more important than speed")
            st.write("**Typical Settings:** N=3-7 candidates, with rotation testing enabled")
            
            st.markdown("#### 🔬 Real-World Impact")
            st.info("**Scenario:** A fossil might appear in Stage 1 as candidate #3 due to lighting/preservation differences, but when Stage 2 applies rotation and detailed metrics, it becomes the clear winner.")
            
            st.markdown("#### ⚡ Processing Workflow Example")
            st.code("""
1. Stage 1 examines 500+ slices, finds scores:
   - Candidate #1: Axial slice 45 (score: 0.85)
   - Candidate #2: Coronal slice 12 (score: 0.83)  
   - Candidate #3: Sagittal slice 67 (score: 0.82)
   - Candidate #4: Axial slice 23 (score: 0.81)
   - Candidate #5: Coronal slice 34 (score: 0.80)

2. Stage 2 fine-tunes each candidate:
   - Candidate #1: Final score 0.87 (no rotation)
   - Candidate #2: Final score 0.91 (rotation 90°)  ← WINNER!
   - Candidate #3: Final score 0.84 (rotation 30°)
   - Candidate #4: Final score 0.86 (no rotation)
   - Candidate #5: Final score 0.83 (rotation 180°)

3. Result: Candidate #2 wins despite being #2 in Stage 1!
            """, language=None)
            
        with st.expander("🧠 Two-Stage Pipeline Technology", expanded=False):
            st.markdown("### 🚀 Two-Stage Coarse-to-Fine Search Algorithm")
            
            st.markdown("#### 🔍 Stage 1: Coarse Search for Initial Pose Estimation")
            st.info("**Goal:** Quickly find the best approximate location of the query slice within a 3D volume")
            st.write("**Metric:** Combines Dice Score + Hu Moments")
            st.write("- **Dice Score:** Measures overlap between binary masks (higher = better)")
            st.write("- **Hu Moments:** Shape descriptors for morphological similarity (distance converted to similarity)")
            st.write("- **IMPROVED:** Hu Moments weight increased from 0.3 to 0.6 for better rotation invariance")
            st.write("- **Combined Score:** `w_dice × dice + w_hu × hu_similarity` (now 0.4 + 0.6)")
            st.write("- **Output:** Initial pose (axis, index) with best coarse score")
            
            st.markdown("#### 🎯 Stage 2: Fine-Tuning with 2D-to-3D Registration")
            st.info("**Goal:** Find precise, sub-slice alignment around the coarse result")
            st.write("**Metric:** Combines Dice Score + NCC + SSIM + ORB Feature Matching")
            st.write("- **Dice Score:** Binary mask overlap for structural alignment")
            st.write("- **SSIM:** Structural similarity for detailed morphological comparison")
            st.write("- **NCC:** Grayscale intensity correlation for texture matching")
            st.write("- **ORB Features:** Rotation-invariant keypoint matching for structural analysis")
            st.write("- **Search:** Local optimization around Stage 1 result")
            st.write("- **NEW:** Optional rotation testing for in-plane rotation invariance")
            st.write("- **Output:** Final pose and optimized similarity score")
            
            st.markdown("#### 🔄 NEW: Rotation Invariance Technology")
            st.success("**Innovation:** In-Plane Rotation Testing in Stage 2")
            st.write("**Method:**")
            st.write("1. For each candidate slice from the 3D volume")
            st.write("2. Test rotations: [0°, 30°, 60°, 90°, 120°, 150°, 180°, 210°, 240°, 270°, 300°, 330°]")
            st.write("3. Calculate fine-tuning score for each rotation")
            st.write("4. Use the maximum score from all rotations as the final score")
            st.write("5. Store the best rotation angle for each slice")
            
            st.markdown("#### 🔑 NEW: ORB Feature Matching Technology")
            st.success("**Innovation:** Oriented FAST and Rotated BRIEF (ORB) for structural analysis")
            st.write("**Benefits:**")
            st.write("- **Patent-Free:** Fast, effective, and royalty-free feature detection")
            st.write("- **Rotation Invariant:** Built-in orientation compensation")
            st.write("- **Scale Robust:** Multi-level pyramid feature detection")
            st.write("- **Structural Focus:** Detects corners and edges critical for fossil identification")
            st.write("**Implementation:**")
            st.write("- FAST keypoint detector with Harris corner response")
            st.write("- Oriented BRIEF descriptors for rotation invariance")
            st.write("- Brute-force matching with Lowe's ratio test")
            st.write("- Normalized scoring based on match quality and quantity")
            
            st.markdown("#### ⚡ Key Advantages")
            st.success("✅ **Improved Accuracy:** Two-stage approach reduces false positives")
            st.success("✅ **Enhanced Robustness:** Four complementary similarity measures")
            st.success("✅ **Better Shape Matching:** Hu moments capture morphological features")
            st.success("✅ **Feature-Based Analysis:** ORB detects structural keypoints")
            st.success("✅ **Rotation Invariance:** Finds matches regardless of fossil orientation")
            st.success("✅ **Consistent Segmentation:** Same processing applied to query and volume slices")
            st.success("✅ **Efficient Search:** Coarse-to-fine reduces computational complexity")
            
            st.markdown("#### 🔬 Technical Implementation")
            st.write("1. **Binary Mask Creation:** Otsu thresholding for fossil segmentation")
            st.write("2. **Shape Analysis:** 7 Hu moment invariants for rotation/scale independence")
            st.write("3. **Local Refinement:** ±5 slice search radius around coarse result")
            st.write("4. **ORB Feature Detection:** Up to 500 keypoints per image with Harris scoring")
            st.write("5. **Rotation Testing:** 12 angles tested per slice using PyTorch affine transformations")
            st.write("6. **Score Fusion:** Weighted combination (Dice: 30%, NCC: 25%, SSIM: 25%, ORB: 20%)")
            
            st.markdown("#### 📊 Performance Impact")
            st.warning("**Computational Cost:** Rotation testing increases processing time by ~12x per slice")
            st.warning("**Memory Usage:** Multiple rotated images stored temporarily")
            st.info("**Quality Benefit:** Dramatically improved accuracy for rotated fossil specimens")
            st.info("**Practical Impact:** Handles real-world scanning variations and mounting orientations")
        
        with st.expander("🧠 Deep Learning Segmentation Technology", expanded=False):
            st.markdown("### 🎯 Pure Fossil-to-Fossil Matching")
            st.success("✨ **Advanced AI Pipeline:** Multi-algorithm consensus approach")
            st.success("🔬 **Technology Stack:** Otsu + Adaptive + Percentile + Histogram analysis")
            st.success("🎯 **Result:** Pure fossil structures without interference")
            
            st.markdown("### ⚡ Performance Benefits")
            st.warning("🚫 **Eliminates:** Background noise, lighting variations, substrate differences")
            st.warning("🔍 **Focuses on:** Fossil morphology, structure, and diagnostic features")
            st.warning("📈 **Accuracy:** Dramatically improves species identification precision")
            
            st.markdown("### 🎚️ How It Works")
            st.info("1️⃣ **Input Analysis:** Applies identical segmentation to your image")
            st.info("2️⃣ **Volume Processing:** Segments all 3D model slices using same algorithms")
            st.info("3️⃣ **Fair Comparison:** Compares pure fossil content vs pure fossil content")
            st.info("4️⃣ **Enhanced Accuracy:** Ignores mounting, lighting, and preservation differences")
            st.markdown("### 🎯 Pure Fossil-to-Fossil Matching")
            st.success("✨ **Advanced AI Pipeline:** Multi-algorithm consensus approach")
            st.success("🔬 **Technology Stack:** Otsu + Adaptive + Percentile + Histogram analysis")
            st.success("🎯 **Result:** Pure fossil structures without interference")
            
            st.markdown("### ⚡ Performance Benefits")
            st.warning("🚫 **Eliminates:** Background noise, lighting variations, substrate differences")
            st.warning("🔍 **Focuses on:** Fossil morphology, structure, and diagnostic features")
            st.warning("📈 **Accuracy:** Dramatically improves species identification precision")
            
            st.markdown("### 🎚️ How It Works")
            st.info("1️⃣ **Input Analysis:** Applies identical segmentation to your image")
            st.info("2️⃣ **Volume Processing:** Segments all 3D model slices using same algorithms")
            st.info("3️⃣ **Fair Comparison:** Compares pure fossil content vs pure fossil content")
            st.info("4️⃣ **Enhanced Accuracy:** Ignores mounting, lighting, and preservation differences")
        
        enable_segmentation = st.checkbox("🚀 Enable Advanced AI Segmentation", value=True,
                                        help="Activate deep learning fossil extraction for maximum accuracy")
        
        # Enhanced segmentation feedback
        if enable_segmentation:
            st.success("🎯 **AI-Enhanced Matching Active** - Advanced neural segmentation will be applied to both your image and all 3D model slices for pure fossil-to-fossil comparison")
        else:
            st.markdown("""
                <div style="padding: 1rem; background: rgba(107, 114, 128, 0.1); border: 1px solid rgba(107, 114, 128, 0.3); border-radius: 8px;">
                    <p style="margin: 0; color: #9ca3af;">� Standard matching mode - using raw image data without segmentation</p>
                </div>
            """, unsafe_allow_html=True)
        
        segmentation_threshold = 0.5
        show_segmentation = False
        preserve_holes = True  # Default to preserving internal structure
        if enable_segmentation:
            segmentation_threshold = st.slider("Segmentation Sensitivity", 0.1, 0.9, 0.5, 0.05,
                                             help="Higher = more selective (only brightest fossil parts). Lower = includes more fossil details.")
            
            preserve_holes = st.checkbox("🏛️ Preserve Internal Structure", value=True,
                                       help="Maintain chambers, pores, and internal holes instead of filling them")
            
            # Add helpful guidance
            if segmentation_threshold < 0.3:
                st.info("🔍 **Low sensitivity:** Includes more fossil details but may include some background")
            elif segmentation_threshold > 0.7:
                st.info("🎯 **High sensitivity:** Very selective - only brightest fossil structures")
            else:
                st.info("⚖️ **Balanced sensitivity:** Good compromise between detail and background removal")
            
            # Structure preservation guidance
            if preserve_holes:
                st.success("🏛️ **Structure Preservation ON:** Internal chambers, pores, and diagnostic holes will be preserved")
            else:
                st.warning("🔄 **Aggressive Mode:** Internal holes will be filled - may reduce species specificity")
        
        # Enhanced preprocessing for cropped fossils
        if enable_segmentation:
            # Apply enhanced fossil segmentation
            st.write("🎯 **Applying Enhanced Fossil Segmentation...**")
            
            # Get the segmented result and mask
            segmented_slice, segmentation_mask = apply_fossil_segmentation(slice_np, segmentation_threshold, 
                                                                       return_mask=True, preserve_holes=preserve_holes)
            
            # Always show the three-image comparison for all processing modes
            st.markdown("### 📸 Segmentation Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if processing_method_used == "original":
                    st.image(original_pil_resized_for_display, caption="📏 Original Image", use_container_width=True)
                elif processing_method_used == "auto_crop":
                    # Show the cropped original
                    original_display = (slice_np - slice_np.min()) / (slice_np.max() - slice_np.min() + 1e-8)
                    st.image(original_display, caption="� Auto-Cropped Original", use_container_width=True)
                elif processing_method_used == "manual_crop":
                    # Show the manually cropped original
                    original_display = (slice_np - slice_np.min()) / (slice_np.max() - slice_np.min() + 1e-8)
                    st.image(original_display, caption="✂️ Manually Cropped Original", use_container_width=True)
            
            with col2:
                # Show the segmentation mask
                mask_display = segmentation_mask.astype(np.float32)
                st.image(mask_display, caption="🎯 Segmentation Mask", use_container_width=True)
            
            with col3:
                # Show the segmented result
                segmented_display = (segmented_slice - segmented_slice.min()) / (segmented_slice.max() - segmented_slice.min() + 1e-8)
                st.image(segmented_display, caption="✨ Segmented Fossil", use_container_width=True)
            
            # Calculate segmentation effectiveness
            original_nonzero = np.sum(slice_np > slice_np.mean())
            segmented_nonzero = np.sum(segmented_slice > 0)
            reduction_percentage = (1 - segmented_nonzero / original_nonzero) * 100 if original_nonzero > 0 else 0
            
            # Show segmentation statistics
            fossil_pixels = np.sum(segmentation_mask)
            total_pixels = segmentation_mask.size
            fossil_percentage = (fossil_pixels / total_pixels) * 100
            
            st.success(f"🧠 **Segmentation Applied:** Reduced image content by {reduction_percentage:.1f}% - "
                      f"focusing on {fossil_pixels:,} fossil pixels ({fossil_percentage:.1f}% of image)")
            
            slice_np = segmented_slice
        else:
            # Even if segmentation is disabled, show the original image for "Use Original" mode
            if processing_method_used == "original":
                st.write("📏 **Original Image Processing**")
                # Show the original PIL image resized to 224x224
                st.image(original_pil_resized_for_display, caption="📏 Original Uploaded (224×224)", use_container_width=True)
        
        # Apply percentile-based normalization (same as dataset creation)
        p1, p99 = np.percentile(slice_np, [1, 99])
        slice_np = np.clip(slice_np, p1, p99)
        
        if p99 > p1:
            slice_np = (slice_np - p1) / (p99 - p1)
        
        # Final normalization
        mi, ma = slice_np.min(), slice_np.max()
        slice_display_np = np.zeros_like(slice_np) if ma-mi<1e-8 else (slice_np-mi)/(ma-mi)
        slice_t = normalize_to_01_torch(torch.from_numpy(slice_np[None,None]).to(device))
        
        # Show final processed image info
        processing_summary = f"📋 **Processing Summary:**\n"
        processing_summary += f"- Image mode: {image_processing_mode}\n"
        
        if processing_method_used == "auto_crop":
            processing_summary += "- Status: ✅ Auto-detected fossil region, cropped after resize (scale preserved)\n"
        elif processing_method_used == "manual_crop":
            processing_summary += "- Status: ✅ Manual selection, cropped after resize (scale preserved)\n"
        else:
            processing_summary += "- Status: 📷 Using original full image, resized to 224×224\n"
        
        if enable_segmentation:
            processing_summary += f"- Segmentation: 🎯 Fossil-only matching (threshold: {segmentation_threshold:.2f})\n"
        
        processing_summary += f"- Final image size: {slice_display_np.shape[1]}×{slice_display_np.shape[0]} pixels\n"
        processing_summary += f"- Method: Resize-first approach preserves fossil scale relationships"
        
        with st.expander("📋 View Processing Details", expanded=False):
            st.markdown(f"""
                <div class="dark-card">
                    <pre>{processing_summary}</pre>
                </div>
            """, unsafe_allow_html=True)
        
        # Store processed image in session state for other tabs
        st.session_state.processed_slice = slice_np
        st.session_state.slice_display_np = slice_display_np
        st.session_state.slice_tensor = slice_t
        st.session_state.enable_segmentation = enable_segmentation
        st.session_state.segmentation_threshold = segmentation_threshold
        st.session_state.processing_method_used = processing_method_used
        
        # Enhanced processing complete section
        st.markdown(f"""
            <div class="success-card" style="margin-top: 2rem;">
                <h3 style="margin: 0 0 1rem 0; display: flex; align-items: center;">
                    <span class="icon">✅</span>Processing Pipeline Complete
                </h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                    <div>
                        <h5 style="margin: 0 0 0.5rem 0; color: rgba(255,255,255,0.9);">Processing Method</h5>
                        <p style="margin: 0; font-weight: 600;">{image_processing_mode}</p>
                    </div>
                    <div>
                        <h5 style="margin: 0 0 0.5rem 0; color: rgba(255,255,255,0.9);">Output Resolution</h5>
                        <p style="margin: 0; font-weight: 600;">224 × 224 pixels</p>
                    </div>
                    <div>
                        <h5 style="margin: 0 0 0.5rem 0; color: rgba(255,255,255,0.9);">AI Segmentation</h5>
                        <p style="margin: 0; font-weight: 600;">{'🚀 Enabled' if enable_segmentation else '🔒 Disabled'}</p>
                    </div>
                    <div>
                        <h5 style="margin: 0 0 0.5rem 0; color: rgba(255,255,255,0.9);">Status</h5>
                        <p style="margin: 0; font-weight: 600;">Ready for Analysis</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 🗂️ Model Selection & Management")
    
    # Model source selection
    st.markdown("#### 🗂️ Model Source")
    model_source = st.radio(
        "Choose model source:",
        ["Upload Your Own Models", "Built-In Models"],
        help="Select whether to use built-in models or upload your own"
    )
    
    st.markdown("---")
    
    if model_source == "Built-In Models":
        # Enhanced model directory handling
        try:
            # Works in normal .py scripts
            PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
        except NameError:
            # Fallback for Jupyter notebooks
            PROJECT_ROOT = pathlib.Path.cwd().parent

        MODELS_DIR=PROJECT_ROOT / "4_Dashboard_App" / "models"
        possible_dirs = [
            "models",
            "h:/3D_Fossil_projects/streamlit_app/models", 
            "/workspace/Fossil_projectc_2316/3d_Fossil_ai_project/models",MODELS_DIR,
            os.path.join(os.path.dirname(__file__), "models")
        ]
        
        model_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                model_dir = dir_path
                break
        
        if model_dir is None:
            st.error("❌ No model directory found. Please check the following locations:")
            for d in possible_dirs:
                st.code(d)
            st.stop()
        
        built_in = [f for f in os.listdir(model_dir) if f.endswith((".nii", ".nii.gz"))]
        
        if not built_in:
            st.error(f"❌ No NIfTI files found in {model_dir}")
            st.stop()
        
        # Enhanced model display with species grouping
        st.markdown(f"""
            <div class="info-card">
                <h4>📁 Model Directory</h4>
                <p><strong>Location:</strong> {model_dir}</p>
                <p><strong>Models Found:</strong> {len(built_in)}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Group models by species
        species_groups = {}
        for model in built_in:
            species = get_species_from_filename(model)
            if species not in species_groups:
                species_groups[species] = []
            species_groups[species].append(model)
        
        # Display species selection
        st.markdown("#### 🦕 Species Selection")
        
        # Add helpful information about the collection
        with st.expander("📊 About This Collection", expanded=False):
            st.markdown(f"""
            **🧬 Comprehensive Fossil Species Database**
            
            This collection contains **{len(built_in)} high-quality 3D models** representing **{len(species_groups)} different fossil species**.
            
            **📈 Collection Statistics:**
            - **Largest species collection:** Chrysalidina ({len(species_groups.get('Chrysalidina', []))} models)
            - **Species with multiple models:** {len([s for s in species_groups if len(species_groups[s]) > 1])} species
            - **Single-specimen species:** {len([s for s in species_groups if len(species_groups[s]) == 1])} species
            
            **🔬 Species Include:**
            - **Foraminifera:** Various benthic and planktonic species
            - **Age Range:** Paleogene to recent specimens  
            - **Imaging:** High-resolution micro-CT reconstructions
            - **Format:** NIfTI medical imaging format (.nii)
            
            **💡 Selection Tips:**
            - **Select All:** Use all models for comprehensive matching
            - **By Species:** Choose specific fossil groups of interest
            - **Individual:** Pick specific models for targeted analysis
            """)
        
        # Option to select all or by species
        selection_mode = st.radio(
            "Selection Mode:",
            ["Select by Species", "Select Individual Models", "Select All"]
        )
        
        if selection_mode == "Select All":
            selected_models = built_in
            st.info(f"✅ All {len(built_in)} models selected")
        
        elif selection_mode == "Select by Species":
            # Display comprehensive species overview first
            st.markdown(f"""
                <div class="info-card">
                    <h4>🧬 Species Collection Overview</h4>
                    <p><strong>Total Species:</strong> {len(species_groups)}</p>
                    <p><strong>Total Models:</strong> {len(built_in)}</p>
                    <p><strong>Single-model species:</strong> {len([s for s in species_groups if len(species_groups[s]) == 1])}</p>
                    <p><strong>Multi-model species:</strong> {len([s for s in species_groups if len(species_groups[s]) > 1])}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Sort species by model count (descending) for better display
            sorted_species = sorted(species_groups.keys(), key=lambda x: len(species_groups[x]), reverse=True)
            
            selected_species = st.multiselect(
                "Choose species to include:",
                sorted_species,
                default=sorted_species[:5] if len(sorted_species) > 5 else sorted_species,
                help="Species are sorted by number of models (most models first)"
            )
            
            if not selected_species:
                st.warning("⚠️ Please select at least one species to continue")
                st.stop()
            
            # Initialize final selected models list
            selected_models = []
            
            # For each selected species, show details and model selection options
            st.markdown("#### 🔍 Species Details & Model Selection")
            
            for species in selected_species:
                species_models = species_groups[species]
                model_count = len(species_models)
                
                with st.expander(f"🦕 {species} ({model_count} models)", expanded=model_count <= 3):
                    
                    # Show species statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                            <div class="species-card">
                                <strong>📊 {species} Statistics</strong><br>
                                <strong>Models:</strong> {model_count}<br>
                                <strong>Category:</strong> {'Single-model' if model_count == 1 else 'Multi-model'}<br>
                                <strong>Percentage of collection:</strong> {(model_count/len(built_in)*100):.1f}%
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Model selection option for this species
                        if model_count == 1:
                            st.info(f"📌 **Single model available** - automatically included")
                            selected_models.extend(species_models)
                        else:
                            model_selection_mode = st.radio(
                                f"Selection for {species}:",
                                ["Use All Models", "Select Specific Models"],
                                key=f"selection_mode_{species}",
                                help=f"Choose how to select from the {model_count} available models"
                            )
                            
                            if model_selection_mode == "Use All Models":
                                selected_models.extend(species_models)
                                st.success(f"✅ All {model_count} models included")
                            else:
                                # Show model selection with better formatting
                                st.markdown(f"**Select specific models from {species}:**")
                                
                                # Create a more user-friendly model display
                                model_display_names = []
                                for model in species_models:
                                    # Create a cleaner display name
                                    if model.startswith("avizo_"):
                                        display_name = model.replace("avizo_", "").replace("_", " ")
                                        # Truncate very long names
                                        if len(display_name) > 60:
                                            display_name = display_name[:57] + "..."
                                    else:
                                        display_name = model.replace("_", " ")
                                        if len(display_name) > 60:
                                            display_name = display_name[:57] + "..."
                                    
                                    model_display_names.append(f"{display_name}")
                                
                                selected_species_models = st.multiselect(
                                    f"Models for {species}:",
                                    options=species_models,
                                    format_func=lambda x: model_display_names[species_models.index(x)],
                                    default=species_models[:3] if len(species_models) > 3 else species_models,
                                    key=f"models_for_{species}",
                                    help=f"Select specific models from {species} collection"
                                )
                                
                                selected_models.extend(selected_species_models)
                                
                                if selected_species_models:
                                    st.success(f"✅ {len(selected_species_models)} out of {model_count} models selected")
                                else:
                                    st.warning(f"⚠️ No models selected for {species}")
                    
                    # Show the actual model filenames with a checkbox toggle
                    show_filenames = st.checkbox(
                        f"📁 Show model filenames for {species}",
                        key=f"show_filenames_{species}",
                        help=f"Display the actual .nii filenames for all {model_count} models"
                    )
                    
                    if show_filenames:
                        st.markdown("**Model Files:**")
                        for i, model in enumerate(species_models, 1):
                            st.code(f"{i}. {model}", language=None)
            
            # Summary of final selection
            if selected_models:
                selected_species_summary = {}
                for model in selected_models:
                    species = get_species_from_filename(model)
                    selected_species_summary[species] = selected_species_summary.get(species, 0) + 1
                
                st.markdown("#### 📋 Final Selection Summary")
                summary_cols = st.columns(min(3, len(selected_species_summary)))
                
                for i, (species, count) in enumerate(selected_species_summary.items()):
                    with summary_cols[i % len(summary_cols)]:
                        st.markdown(f"""
                            <div class="success-card">
                                <strong>🦕 {species}</strong><br>
                                {count} model{'s' if count != 1 else ''}
                            </div>
                        """, unsafe_allow_html=True)
                
                st.success(f"🎯 **Total Selection:** {len(selected_models)} models from {len(selected_species_summary)} species")
            else:
                st.error("❌ No models selected. Please select at least one model from the chosen species.")
        
        else:  # Individual selection
            selected_models = st.multiselect(
                "Choose individual models:",
                built_in,
                default=built_in[:10] if len(built_in) > 10 else built_in
            )
        
        def get_path(name): 
            return os.path.join(model_dir, name)
    
    else:
        # Upload custom models
        st.markdown("#### 📤 Upload Custom Models")
        
        uploads = st.file_uploader(
            "Upload NIfTI model files (.nii, .nii.gz)",
            type=["nii", "nii.gz"],
            accept_multiple_files=True
        )
        
        if not uploads:
            st.info("📁 Upload model files to continue")
            st.stop()
        
        tmp_dir = os.path.join(tempfile.gettempdir(), "custom_models")
        os.makedirs(tmp_dir, exist_ok=True)
        
        selected_models = []
        for up in uploads:
            fp = os.path.join(tmp_dir, up.name)
            with open(fp, "wb") as f:
                f.write(up.getbuffer())
            selected_models.append(up.name)
        
        def get_path(name): 
            return os.path.join(tmp_dir, name)
        
        st.success(f"✅ {len(selected_models)} custom models uploaded")

with tab3:
    st.markdown("### 🎯 Matching Process & Results")
    
    # Multi-Candidate Pipeline weights configuration
    st.markdown("#### 🏆 Multi-Candidate Pipeline Weights")
    st.markdown("""
        <div style="padding: 0.5rem; background: rgba(139, 69, 19, 0.1); border: 1px solid rgba(139, 69, 19, 0.3); border-radius: 6px; margin-bottom: 1rem;">
            <small style="color: #d2691e;">
                <strong>Stage 2 Fine-Tuning Metrics:</strong> Configure the weights for the four-metric scoring system used in Stage 2 fine-tuning phase of the Multi-Candidate Pipeline.
            </small>
        </div>
    """, unsafe_allow_html=True)
    
    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    
    with col_w1:
        w_dice_fine = st.slider("Dice Weight", 0.0, 1.0, 0.3, 0.05,
                               help="Weight for Dice coefficient in Stage 2 fine-tuning. Measures overlap similarity.")
    
    with col_w2:
        w_ncc_fine = st.slider("NCC Weight", 0.0, 1.0, 0.25, 0.05,
                              help="Weight for Normalized Cross-Correlation in Stage 2. Measures intensity correlation.")
    
    with col_w3:
        w_ssim_fine = st.slider("SSIM Weight", 0.0, 1.0, 0.25, 0.05,
                               help="Weight for Structural Similarity Index in Stage 2. Measures structural patterns.")
    
    with col_w4:
        w_orb_fine = st.slider("ORB Weight", 0.0, 1.0, 0.2, 0.05,
                              help="Weight for ORB feature matching in Stage 2. Measures keypoint similarity.")
    
    # Show total weight and normalization
    total_weight = w_dice_fine + w_ncc_fine + w_ssim_fine + w_orb_fine
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.metric("Total Weight", f"{total_weight:.2f}", 
                 help="Sum of all weights. Will be automatically normalized to 1.0 during computation.")
    
    with col_info2:
        if total_weight > 0:
            st.metric("Auto-Normalized", "✓ Enabled", 
                     help="Weights are automatically normalized so they sum to 1.0")
        else:
            st.warning("⚠️ Total weight cannot be zero")
    
    # Add explanation of the current weight distribution
    if total_weight > 0:
        dice_pct = (w_dice_fine / total_weight) * 100
        ncc_pct = (w_ncc_fine / total_weight) * 100
        ssim_pct = (w_ssim_fine / total_weight) * 100
        orb_pct = (w_orb_fine / total_weight) * 100
        
        if dice_pct > 40:
            st.info(f"� **Overlap-focused:** {dice_pct:.1f}% Dice weight emphasizes geometric overlap")
        elif orb_pct > 30:
            st.info(f"� **Feature-focused:** {orb_pct:.1f}% ORB weight emphasizes keypoint matching")
        else:
            st.info(f"⚖️ **Balanced approach:** Dice {dice_pct:.1f}%, NCC {ncc_pct:.1f}%, SSIM {ssim_pct:.1f}%, ORB {orb_pct:.1f}%")
    
    # Multi-candidate configuration
    st.markdown("#### 🎯 Multi-Candidate Configuration")
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        top_n_candidates = st.slider(
            "🏆 Number of Candidates", 
            min_value=3, max_value=10, value=5, step=1,
            help="How many top candidates from Stage 1 should be processed in Stage 2"
        )
    
    with col_c2:
        st.metric("Rotation Testing", "Always Enabled", 
                 help="12 rotations (0°-330°) tested per candidate for maximum accuracy")
    
    # Configuration change detection and auto-clear results
    current_config = {
        'w_dice_fine': w_dice_fine,
        'w_ncc_fine': w_ncc_fine, 
        'w_ssim_fine': w_ssim_fine,
        'w_orb_fine': w_orb_fine,
        'top_n_candidates': top_n_candidates,
        'enable_diag': enable_diag,
        'axes_options': axes_options,
        'selected_models': len(selected_models) if selected_models else 0,
        'segmentation_enabled': st.session_state.get('enable_segmentation', False)
    }
    
    # Check if configuration has changed
    if 'last_config' not in st.session_state:
        st.session_state.last_config = current_config
    elif st.session_state.last_config != current_config:
        # Configuration changed - clear previous results
        st.session_state.matching_results = []
        st.session_state.best_match_info = None
        st.session_state.last_config = current_config
        st.session_state.config_changed = True  # Flag to show notification
        st.rerun()  # Refresh to show cleared results
    
    # Show notification if results were cleared due to config change
    if st.session_state.get('config_changed', False):
        st.info("🔄 **Configuration changed** - Previous results cleared. Click 'Start Matching Process' to run with new settings.")
        st.session_state.config_changed = False  # Reset flag
    
    st.markdown("---")
    
    # Check if we have required data
    if 'processed_slice' not in st.session_state:
        st.warning("⚠️ Please upload and process an image in the 'Upload & Process' tab first")
        st.stop()
    
    if not selected_models:
        st.warning("⚠️ Please select models in the 'Model Selection' tab first")
        st.stop()
    
    # Enhanced matching configuration display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚙️ Current Configuration")
        
        # Get the actual segmentation setting from upload processing
        segmentation_enabled = st.session_state.get('enable_segmentation', False)
        
        # Calculate normalized weights for display
        total_weight = w_dice_fine + w_ncc_fine + w_ssim_fine + w_orb_fine
        if total_weight > 0:
            dice_norm = w_dice_fine / total_weight
            ncc_norm = w_ncc_fine / total_weight
            ssim_norm = w_ssim_fine / total_weight
            orb_norm = w_orb_fine / total_weight
        else:
            dice_norm = ncc_norm = ssim_norm = orb_norm = 0.25
        
        config_info = f"""
        **🔍 Search Axes:** {axes_options}  
        **🚀 Pipeline:** Multi-Candidate Fine-Tuning
        
        **🎯 Image Segmentation:** {'Enabled' if segmentation_enabled else 'Disabled'}  
        **🏆 Candidates:** Top {top_n_candidates} processed  
        **🔄 Rotation:** Multi-Candidate + Rotation (Always Enabled)  
        **⚖️ Stage 2 Weights:**  
        └── Dice: {dice_norm:.2f} ({dice_norm*100:.1f}%)  
        └── NCC: {ncc_norm:.2f} ({ncc_norm*100:.1f}%)  
        └── SSIM: {ssim_norm:.2f} ({ssim_norm*100:.1f}%)  
        └── ORB: {orb_norm:.2f} ({orb_norm*100:.1f}%)  
        **🗂️ Models:** {len(selected_models)} selected  
        **↗️ Diagonal Search:** {'Enabled' if enable_diag else 'Disabled'} {''if enable_diag else ''}
        """
        st.markdown(config_info)
    
    with col2:
        st.markdown("#### 🚀 Start Matching")
        
        if st.button("🔍 Start Matching Process", type="primary", use_container_width=True):
            # Clear previous results
            st.session_state.matching_results = []
            st.session_state.best_match_info = None
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_tracker = st.empty()
            start_time = time.time()
            
            best_overall = -999
            
            # Process each model
            for i, model_name in enumerate(selected_models):
                try:
                    current_time = time.time() - start_time
                    status_text.markdown(f"""
                        <div class="info-card">
                            <h5>Processing: {model_name}</h5>
                            <p>Model {i+1} of {len(selected_models)} | Time: {current_time:.1f}s</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Load and preprocess volume
                    nii = nib.load(get_path(model_name))
                    vol_raw = nii.get_fdata().astype(np.float32)
                    
                    if vol_raw.size == 0:
                        st.error(f"❌ Empty volume in {model_name}")
                        continue
                    
                    # Apply preprocessing (same as original)
                    vol = resample_isotropic(vol_raw, nii.header)
                    
                    # Check for valid volume after resampling
                    if vol.size == 0:
                        st.error(f"❌ Volume became empty after resampling: {model_name}")
                        continue
                    
                    vmin, vmax = volume_minmax_percentile(vol)
                    vol_normalized = np.clip(vol, vmin, vmax)
                    
                    if vmax > vmin:
                        vol_normalized = (vol_normalized - vmin) / (vmax - vmin)
                    
                    # Additional safety check
                    if np.isnan(vol_normalized).any() or np.isinf(vol_normalized).any():
                        st.error(f"❌ Invalid values in normalized volume: {model_name}")
                        continue
                    
                    st.session_state.loaded_volumes[model_name] = vol_normalized
                    
                    # Convert to torch
                    vol_t = torch.from_numpy(vol_normalized[None, None]).float().to(device)
                    vol_t = normalize_to_01_torch(vol_t)
                    
                    # Get processed slice from session state
                    if 'slice_tensor' in st.session_state:
                        # Use the stored tensor if available
                        slice_t = st.session_state.slice_tensor.to(device)
                    elif 'processed_slice' in st.session_state:
                        slice_np = st.session_state.processed_slice
                        
                        # Convert slice to torch tensor with proper dimensions
                        if isinstance(slice_np, np.ndarray):
                            if slice_np.ndim == 2:
                                # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
                                slice_t = torch.from_numpy(slice_np[None, None]).float().to(device)
                            elif slice_np.ndim == 3:
                                # Add batch dimension: (C, H, W) -> (1, C, H, W)
                                slice_t = torch.from_numpy(slice_np[None]).float().to(device)
                            else:
                                # Already has batch dimension
                                slice_t = torch.from_numpy(slice_np).float().to(device)
                        else:
                            slice_t = slice_np.to(device)  # Already a tensor
                        
                        # Normalize slice to [0,1]
                        slice_t = normalize_to_01_torch(slice_t)
                    else:
                        st.error("❌ No processed slice found in session state")
                        continue
                    
                    # Debug information
                    # st.write(f"🔍 Slice tensor shape: {slice_t.shape}")
                    # st.write(f"🔍 Volume tensor shape: {vol_t.shape}")
                    
                    # Run actual matching using the restored functions
                    # Use the same segmentation setting as was applied to the input image
                    use_segmentation_for_matching = st.session_state.get('enable_segmentation', False)
                    segmentation_threshold_for_matching = st.session_state.get('segmentation_threshold', 0.5)
                    
                    # Determine angle_list for diagonal search (use Preset Optimal mode)
                    angle_list = ROTATION_ANGLES if enable_diag else None
                    
                    match_result = match_slice_to_volume_torch(
                        volume_t=vol_t,
                        slice_t=slice_t,
                        axes=axes_options,
                        enable_diagonal=enable_diag,
                        angle_step=30,
                        use_all_degrees=False,
                        coarse_to_fine=True,
                        angle_list=angle_list,  # NEW: Use scientifically optimized angles for diagonal search
                        fossil_area_threshold=0.05,
                        use_segmentation=use_segmentation_for_matching,
                        segmentation_threshold=segmentation_threshold_for_matching,
                        use_two_stage=use_two_stage,  # Use two-stage pipeline
                        enable_rotation_testing=enable_rotation_testing,  # Enable rotation invariance
                        preserve_holes=preserve_holes,  # Preserve internal structure
                        use_multi_candidate=use_multi_candidate,  # NEW: Enable multi-candidate approach
                        top_n_candidates=top_n_candidates,  # NEW: Number of candidates to process
                        w_dice_fine=w_dice_fine, w_ncc_fine=w_ncc_fine, w_ssim_fine=w_ssim_fine, w_orb_fine=w_orb_fine  # NEW: Fine-tuning weights
                    )
                    
                    # Validate match result
                    if not match_result or 'score' not in match_result:
                        st.warning(f"⚠️ No valid match found for {model_name}")
                        continue
                    
                    # Store the complete result
                    match_result["model"] = model_name
                    match_result["species"] = get_species_from_filename(model_name)
                    
                    if match_result["score"] > best_overall:
                        best_overall = match_result["score"]
                        st.session_state.best_match_info = match_result
                    
                    st.session_state.matching_results.append(match_result)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {model_name}: {str(e)}")
                    # Show more detailed error info for debugging
                    import traceback
                    st.code(traceback.format_exc())
                    continue
                
                progress_bar.progress(int((i + 1) / len(selected_models) * 100))
            
            total_time = time.time() - start_time
            status_text.markdown(f"""
                <div class="success-card">
                    <h5>✅ Matching Complete!</h5>
                    <p>Processed {len(selected_models)} models in {total_time:.1f} seconds</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.session_state.matching_done = True
    
    # Display results if matching is done
    if st.session_state.matching_done and st.session_state.matching_results:
        st.markdown("---")
        st.markdown("### 📋 Matching Results")
        
        # Sort results by score
        scores_sorted = sorted(st.session_state.matching_results, key=lambda x: x["score"], reverse=True)
        
        # Top results display
        st.markdown("#### 🏆 Top Matches")
    
    elif not st.session_state.get('matching_results', []) and st.session_state.get('matching_done', False):
        st.markdown("---")
        st.info("ℹ️ **No matching results available.** Click 'Start Matching Process' to begin analysis.")
    
    elif not st.session_state.get('matching_results', []):
        st.markdown("---")
        st.markdown("### 📋 Matching Results")
        st.info("📝 **Ready to start matching.** Configure your settings above and click 'Start Matching Process' to begin.")
    
    # Show results if available
    if st.session_state.matching_done and st.session_state.matching_results:
        scores_sorted = sorted(st.session_state.matching_results, key=lambda x: x["score"], reverse=True)
        
        for i, result in enumerate(scores_sorted[:5]):
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            species = result.get('species', get_species_from_filename(result['model']))

            
            # Format metrics based on pipeline type
            pipeline_mode = result.get('mode', 'unknown')
            
            if pipeline_mode in ['multi_candidate_two_stage', 'multi_candidate_rotation_invariant']:
                metrics_info = f"<p><strong>Final Score:</strong> {result['score']:.4f}</p>"
                # Include ORB in metrics display if available (show even if 0)
                orb_display = f" | <strong>ORB:</strong> {result.get('orb', 0):.4f}" if 'orb' in result else ""
                metrics_info += f"<p><strong>Dice:</strong> {result.get('dice', 0):.4f} | <strong>NCC:</strong> {result.get('ncc', 0):.4f} | <strong>SSIM:</strong> {result.get('ssim', 0):.4f}{orb_display}</p>"
                
                # Show multi-candidate specific information
                winning_candidate = result.get('winning_candidate', 'N/A')
                total_candidates = result.get('total_candidates_processed', 'N/A')
                metrics_info += f"<p><strong>Winner:</strong> Candidate #{winning_candidate} of {total_candidates}</p>"
                
                # Show rotation information if available
                if result.get('best_rotation', 0) != 0:
                    metrics_info += f"<p><strong>Rotation Applied:</strong> {result['best_rotation']}°</p>"
                
                if pipeline_mode == 'multi_candidate_rotation_invariant':
                    metrics_info += f"<p><strong>Pipeline:</strong> Multi-Candidate + Rotation Invariant</p>"
                else:
                    metrics_info += f"<p><strong>Pipeline:</strong> Multi-Candidate Two-Stage</p>"
            
            elif pipeline_mode in ['two_stage', 'two_stage_rotation_invariant']:
                metrics_info = f"<p><strong>Final Score:</strong> {result['score']:.4f}</p>"
                # Include ORB in metrics display if available (show even if 0)
                orb_display = f" | <strong>ORB:</strong> {result.get('orb', 0):.4f}" if 'orb' in result else ""
                metrics_info += f"<p><strong>Dice:</strong> {result.get('dice', 0):.4f} | <strong>NCC:</strong> {result.get('ncc', 0):.4f} | <strong>SSIM:</strong> {result.get('ssim', 0):.4f}{orb_display}</p>"
                
                # Show rotation information if available
                if result.get('best_rotation', 0) != 0:
                    metrics_info += f"<p><strong>Rotation Applied:</strong> {result['best_rotation']}°</p>"
                
                if pipeline_mode == 'two_stage_rotation_invariant':
                    metrics_info += f"<p><strong>Pipeline:</strong> Two-Stage + Rotation Invariant</p>"
                else:
                    metrics_info += f"<p><strong>Pipeline:</strong> Two-Stage Coarse-to-Fine</p>"
            else:
                metrics_info = f"<p><strong>Combined Score:</strong> {result['score']:.4f}</p>"
                metrics_info += f"<p><strong>SSIM:</strong> {result.get('ssim', 0):.4f} | <strong>NCC:</strong> {result.get('ncc', 0):.4f}</p>"
                metrics_info += f"<p><strong>Pipeline:</strong> Single-Stage</p>"
            
            st.markdown(f"""
                <div class="match-result">
                    <h5>{rank_emoji} {species}</h5>
                    <p><strong>Model:</strong> {result['model']}</p>
                    {metrics_info}
                    <p><strong>Mode:</strong> {result['mode']}</p>
                    {f"<p><strong>Axis:</strong> {result['axis']} | <strong>Index:</strong> {result['index']}</p>" if result.get('axis') is not None else ""}
                    {f"<p><strong>Angles:</strong> {result['angles']}</p>" if result.get('angles') else ""}
                </div>
            """, unsafe_allow_html=True)
        
        # Score threshold filter
        st.markdown("#### 🎯 Filter Results")
        score_threshold = st.slider("Minimum Score Threshold", 0.0, 1.0, 0.85, 0.01)
        
        filtered_results = [r for r in scores_sorted if r["score"] >= score_threshold]
        
        if filtered_results:
            st.success(f"✅ {len(filtered_results)} models meet the threshold")
            
            # Best filtered result
            best_filtered = filtered_results[0]
            species = get_species_from_filename(best_filtered['model'])
            
            st.markdown(f"""
                <div class="success-card">
                    <h4>🎯 Best Match: {species}</h4>
                    <p><strong>Model:</strong> {best_filtered['model']}</p>
                    <p><strong>Score:</strong> {best_filtered['score']:.4f}</p>
                    <p><strong>Match Type:</strong> {best_filtered['mode'].title()}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display final comparison: uploaded image vs best match
            st.markdown("---")
            st.markdown("### 🔍 Final Match Comparison")
            
            # Get the best matched slice (this should already be properly processed from matching)
            best_slice = best_filtered["slice_2d"].cpu().numpy()[0, 0]
            
            # Simply normalize the matched slice for display
            # (it should already be segmented if segmentation was used during matching)
            mi2, ma2 = best_slice.min(), best_slice.max()
            if ma2 - mi2 > 1e-8:
                best_slice_display = (best_slice - mi2) / (ma2 - mi2)
            else:
                best_slice_display = np.zeros_like(best_slice)
            
            # Check if segmentation was used for labeling
            enable_segmentation = st.session_state.get('enable_segmentation', False)
            match_type_label = "🎯 Best Match (Segmented)" if enable_segmentation else "🎯 Best Match"
            
            # Create detailed match description
            fossil_info = ""
            if 'fossil_bbox' in best_filtered and best_filtered['fossil_bbox']:
                x0, x1, y0, y1, z0, z1 = best_filtered['fossil_bbox']
                fossil_size = (x1-x0+1, y1-y0+1, z1-z0+1)
                fossil_info = f" | Fossil Region: {fossil_size[0]}×{fossil_size[1]}×{fossil_size[2]} voxels"
            
            # Generate match details based on mode
            pipeline_mode = best_filtered.get("mode", "unknown")
            
            if pipeline_mode in ['multi_candidate_two_stage', 'multi_candidate_rotation_invariant']:
                winning_candidate = best_filtered.get('winning_candidate', 'N/A')
                total_candidates = best_filtered.get('total_candidates_processed', 'N/A')
                rotation_info = f", rotation: {best_filtered.get('best_rotation', 0)}°" if best_filtered.get('best_rotation', 0) != 0 else ", no rotation needed"
                details = f"Multi-Candidate Winner #{winning_candidate}/{total_candidates}: axis={best_filtered['axis']}, index={best_filtered['index']}{rotation_info}{fossil_info}"
            elif pipeline_mode == "orthogonal":
                details = f"Orthogonal axis={best_filtered['axis']}, index={best_filtered['index']}{fossil_info}"
            elif pipeline_mode == "heavy_rotation":
                axis_names = {(0,2): "XZ rotation", (0,1): "XY rotation", (1,2): "YZ rotation"}
                axis_name = axis_names.get(best_filtered.get('rotation_axis'), 'Unknown axis')
                details = f"Heavy Rotation: {best_filtered.get('rotation_angle', 'Unknown')}° {axis_name}, slice {best_filtered.get('slice_index', 'Unknown')}{fossil_info}"
            elif pipeline_mode == "diagonal":
                details = f"Diagonal angles={best_filtered['angles']}"
            elif pipeline_mode == "two_stage_rotation_invariant":
                rotation_info = f", rotation: {best_filtered.get('best_rotation', 0)}°" if best_filtered.get('best_rotation', 0) != 0 else ", no rotation needed"
                details = f"Two-Stage + Rotation: axis={best_filtered['axis']}, index={best_filtered['index']}{rotation_info}{fossil_info}"
            elif pipeline_mode == "two_stage":
                details = f"Two-Stage: axis={best_filtered['axis']}, index={best_filtered['index']}{fossil_info}"
            else:
                details = f"Mode: {pipeline_mode}"
            
            # Display images side by side
            col1, col2 = st.columns(2)
            
            with col1:
                # Get the uploaded slice for display
                if 'slice_display_np' in st.session_state:
                    uploaded_display = st.session_state.slice_display_np
                else:
                    # Fallback: use processed slice
                    uploaded_slice = st.session_state.processed_slice
                    mi, ma = uploaded_slice.min(), uploaded_slice.max()
                    uploaded_display = np.zeros_like(uploaded_slice) if ma - mi < 1e-8 else (uploaded_slice - mi) / (ma - mi)
                
                # Create upload caption based on segmentation status
                upload_caption = "📤 Your Uploaded Slice"
                if enable_segmentation:
                    upload_caption += " (Segmented)"
                
                st.image(uploaded_display, caption=upload_caption, use_container_width=True)
                
                # Show upload info
                processing_method = st.session_state.get('processing_method_used', 'unknown')
                segmentation_info = " + Segmentation" if enable_segmentation else ""
                st.info(f"**Processing:** {processing_method.replace('_', ' ').title()}{segmentation_info}")
            
            with col2:
                st.image(best_slice_display, 
                        caption=f"{match_type_label}: {best_filtered['model']}\n{details}", 
                        use_container_width=True)
                
                # Show match statistics
                col2a, col2b = st.columns(2)
                with col2a:
                    pipeline_mode = best_filtered.get('mode', '')
                    if pipeline_mode.startswith('multi_candidate') or pipeline_mode.startswith('two_stage'):
                        # Enhanced pipelines with all four metrics
                        st.metric("Dice Score", f"{best_filtered.get('dice', 0):.4f}")
                        st.metric("Final Score", f"{best_filtered['score']:.4f}")
                        if 'orb' in best_filtered:
                            st.metric("ORB Features", f"{best_filtered.get('orb', 0):.4f}")
                    else:
                        # Single-stage pipeline (legacy)
                        st.metric("SSIM", f"{best_filtered.get('ssim', 0):.4f}")
                        st.metric("Combined", f"{best_filtered['score']:.4f}")
                with col2b:
                    st.metric("NCC", f"{best_filtered.get('ncc', 0):.4f}")
                    if 'ssim' in best_filtered and best_filtered.get('ssim', 0) > 0:
                        st.metric("SSIM", f"{best_filtered.get('ssim', 0):.4f}")
                    if best_filtered.get('best_rotation', 0) != 0:
                        st.metric("Rotation", f"{best_filtered['best_rotation']}°")
                    elif 'orb_matches' in best_filtered:
                        st.metric("ORB Matches", f"{best_filtered.get('orb_matches', 0)}")
                    else:
                        st.metric("Species", species)
            
            # Match quality assessment
            score = best_filtered['score']
            if score >= 0.9:
                st.success("🎯 **Excellent Match!** Very high confidence in species identification")
            elif score >= 0.8:
                st.success("✅ **Good Match!** High confidence in species identification")
            elif score >= 0.7:
                st.warning("⚠️ **Moderate Match** - Consider examining multiple top results")
            else:
                st.error("❌ **Low Match Quality** - Results may not be reliable")
            
            # Top Results Explorer - Now Always Available
            st.markdown("---")
            st.markdown("### 🔍 **Top Results Explorer**")
            
            if score >= 0.8:
                st.info("Explore the top candidates to understand the matching process and verify the identification.")
            elif score >= 0.7:
                st.info("Since this is a moderate match, examine the top candidates to make an informed decision.")
            else:
                st.warning("Results have low confidence scores. Use this explorer to find potential better matches.")
            
            # Show top N results for comparison
            num_top_results = min(5, len(filtered_results))
            
            if num_top_results > 1:
                    # Create tabs for each top result
                    tab_labels = []
                    for i, result in enumerate(filtered_results[:num_top_results]):
                        species = get_species_from_filename(result['model'])
                        score_str = f"{result['score']:.3f}"
                        rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
                        tab_labels.append(f"{rank_emoji} {species} ({score_str})")
                    
                    # Create tabs for comparison
                    result_tabs = st.tabs(tab_labels)
                    
                    for i, (tab, result) in enumerate(zip(result_tabs, filtered_results[:num_top_results])):
                        with tab:
                            species = get_species_from_filename(result['model'])
                            
                            # Display result information
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Get the matched slice for this result
                                if 'slice_2d' in result and result['slice_2d'] is not None:
                                    matched_slice = result["slice_2d"].cpu().numpy()[0, 0]
                                    
                                    # Normalize for display
                                    mi, ma = matched_slice.min(), matched_slice.max()
                                    if ma - mi > 1e-8:
                                        matched_slice_display = (matched_slice - mi) / (ma - mi)
                                    else:
                                        matched_slice_display = np.zeros_like(matched_slice)
                                    
                                    # Create comparison display
                                    fig_cols = st.columns(2)
                                    
                                    with fig_cols[0]:
                                        # Show uploaded slice again for comparison
                                        uploaded_display = st.session_state.slice_display_np
                                        st.image(uploaded_display, 
                                                caption="📤 Your Uploaded Slice", 
                                                use_container_width=True)
                                    
                                    with fig_cols[1]:
                                        # Show this result's matched slice
                                        pipeline_mode = result.get('mode', 'unknown')
                                        if pipeline_mode.startswith('multi_candidate'):
                                            pipeline_info = "Multi-Candidate"
                                        elif pipeline_mode.startswith('two_stage'):
                                            pipeline_info = "Two-Stage"
                                        else:
                                            pipeline_info = "Single-Stage"
                                        
                                        st.image(matched_slice_display, 
                                                caption=f"🎯 Match from {species}\n({pipeline_info})", 
                                                use_container_width=True)
                                else:
                                    st.warning("⚠️ No slice image available for this result")
                            
                            with col2:
                                # Show detailed metrics for this result
                                st.markdown(f"""
                                    <div class="metric-card">
                                        <h4>📊 Match Details</h4>
                                        <p><strong>Species:</strong> {species}</p>
                                        <p><strong>Model:</strong> {result['model']}</p>
                                        <p><strong>Score:</strong> {result['score']:.4f}</p>
                                        <hr>
                                """, unsafe_allow_html=True)
                                
                                # Show appropriate metrics based on pipeline type
                                pipeline_mode = result.get('mode', 'unknown')
                                
                                if pipeline_mode in ['multi_candidate_two_stage', 'multi_candidate_rotation_invariant', 'two_stage', 'two_stage_rotation_invariant']:
                                    # Multi-candidate or two-stage pipelines
                                    if pipeline_mode.startswith('multi_candidate'):
                                        pipeline_display = f"Multi-Candidate ({pipeline_mode.replace('multi_candidate_', '').replace('_', ' ').title()})"
                                        
                                        # Show multi-candidate specific info
                                        winning_candidate = result.get('winning_candidate', 'N/A')
                                        total_candidates = result.get('total_candidates_processed', 'N/A')
                                        pipeline_info_extra = f"<p><strong>Winner:</strong> Candidate #{winning_candidate} of {total_candidates}</p>"
                                    else:
                                        pipeline_display = result.get('mode', 'Unknown').replace('_', ' ').title()
                                        pipeline_info_extra = ""
                                    
                                    st.markdown(f"""
                                        <p><strong>Pipeline:</strong> {pipeline_display}</p>
                                        {pipeline_info_extra}
                                        <p><strong>Dice Score:</strong> {result.get('dice', 0):.4f}</p>
                                        <p><strong>NCC:</strong> {result.get('ncc', 0):.4f}</p>
                                        <p><strong>SSIM:</strong> {result.get('ssim', 0):.4f}</p>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show ORB feature matching if available (show even if 0)
                                    if 'orb' in result:
                                        orb_matches = result.get('orb_matches', 0)
                                        st.markdown(f"""
                                            <p><strong>ORB Features:</strong> {result.get('orb', 0):.4f}</p>
                                            <p><strong>Feature Matches:</strong> {orb_matches}</p>
                                        """, unsafe_allow_html=True)
                                    
                                    # Show rotation information if available
                                    if result.get('best_rotation', 0) != 0:
                                        st.markdown(f"""
                                            <p><strong>Rotation Applied:</strong> {result['best_rotation']}°</p>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                        <p><strong>Pipeline:</strong> Single-Stage</p>
                                        <p><strong>SSIM:</strong> {result.get('ssim', 0):.4f}</p>
                                        <p><strong>NCC:</strong> {result.get('ncc', 0):.4f}</p>
                                    """, unsafe_allow_html=True)
                                
                                # Show pose information
                                if result.get('axis') is not None:
                                    axis_name = ["Axial", "Coronal", "Sagittal"][result['axis']]
                                    st.markdown(f"""
                                        <p><strong>Orientation:</strong> {axis_name}</p>
                                        <p><strong>Slice Index:</strong> {result.get('index', 'N/A')}</p>
                                    """, unsafe_allow_html=True)
                                elif result.get('angles'):
                                    st.markdown(f"""
                                        <p><strong>Type:</strong> Diagonal</p>
                                        <p><strong>Angles:</strong> {result['angles']}</p>
                                    """, unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                                
                                # Quality assessment for this specific result
                                result_score = result['score']
                                if result_score >= 0.85:
                                    st.success("✅ **Strong candidate**")
                                elif result_score >= 0.75:
                                    st.warning("⚠️ **Moderate confidence**")
                                else:
                                    st.error("❌ **Low confidence**")
                    
                    # Add comparison guidance
                    st.markdown("---")
                    st.markdown("### 🎯 **Analysis Guidance**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                            **🔍 What to Look For:**
                            - **Structural Similarity**: Compare fossil shapes and features
                            - **Size Proportions**: Check if overall dimensions match
                            - **Key Features**: Look for diagnostic characteristics
                            - **Surface Textures**: Compare surface patterns and details
                        """)
                    
                    with col2:
                        st.markdown("""
                            **📊 Score Interpretation:**
                            - **> 0.85**: Very likely correct identification
                            - **0.75-0.85**: Good candidate, examine visually
                            - **0.70-0.75**: Possible match, compare with others
                            - **< 0.70**: Less likely, but worth considering
                        """)
                    
                    # Species summary for moderate matches
                    unique_species_in_top = list(set([get_species_from_filename(r['model']) for r in filtered_results[:num_top_results]]))
                    
                    if len(unique_species_in_top) > 1:
                        st.info(f"**🧬 Multiple Species Detected:** {len(unique_species_in_top)} different species in top {num_top_results} results: {', '.join(unique_species_in_top)}")
                        
                        # Add a decision helper for multiple species
                        st.markdown("### 🤔 **Decision Helper**")
                        
                        # Calculate species-level scores
                        species_scores = {}
                        species_counts = {}
                        for result in filtered_results[:num_top_results]:
                            species = get_species_from_filename(result['model'])
                            if species not in species_scores:
                                species_scores[species] = []
                                species_counts[species] = 0
                            species_scores[species].append(result['score'])
                            species_counts[species] += 1
                        
                        # Display species comparison
                        st.markdown("**Species-Level Analysis:**")
                        for species in unique_species_in_top:
                            scores = species_scores[species]
                            avg_score = np.mean(scores)
                            max_score = np.max(scores)
                            count = species_counts[species]
                            
                            st.markdown(f"""
                                <div style="padding: 0.5rem; margin: 0.25rem 0; border-left: 3px solid #60a5fa; background: rgba(96, 165, 250, 0.1);">
                                    <strong>🦕 {species}</strong><br>
                                    <small>Models in top {num_top_results}: {count} | Avg Score: {avg_score:.4f} | Best Score: {max_score:.4f}</small>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Recommendation
                        best_species = max(unique_species_in_top, key=lambda s: np.mean(species_scores[s]))
                        st.success(f"**🎯 Recommendation:** Based on average scores, **{best_species}** appears most likely.")
                    else:
                        st.success(f"**🧬 Consistent Species:** All top {num_top_results} results point to **{unique_species_in_top[0]}**")
                        st.info("Multiple models of the same species show similar matches - this increases confidence in the identification.")
                
            else:
                st.info("Only one result meets the threshold criteria.")
        
        else:
            st.info("Only one result available for comparison.")
            
    else:
        st.warning("⚠️ No models meet the current threshold")

with tab4:
    st.markdown("### 📊 Analysis & Statistics")
    
    if not st.session_state.matching_done:
        st.info("📊 Complete matching process to view detailed analysis")
    else:
        # Species analysis
        st.markdown("#### 🦕 Species Analysis")
        
        # Group results by species
        species_results = {}
        for result in st.session_state.matching_results:
            species = get_species_from_filename(result['model'])
            if species not in species_results:
                species_results[species] = []
            species_results[species].append(result['score'])
        
        # Calculate species statistics
        species_stats = {}
        for species, scores in species_results.items():
            species_stats[species] = {
                'avg_score': np.mean(scores),
                'max_score': np.max(scores),
                'min_score': np.min(scores),
                'std_score': np.std(scores),
                'model_count': len(scores)
            }
        
        # Sort by average score
        species_ranked = sorted(species_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        
        # Display species ranking
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🏆 Species Ranking by Average Score")
            for i, (species, stats) in enumerate(species_ranked):
                rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}."
                st.markdown(f"""
                    <div class="species-card">
                        <strong>{rank_emoji} {species}</strong><br>
                        Avg: {stats['avg_score']:.4f} | Max: {stats['max_score']:.4f}<br>
                        Models: {stats['model_count']}
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Create score distribution chart
            valid_results = [r for r in st.session_state.matching_results if r.get('score', 0) > 0]
            if valid_results:
                scores = [r['score'] for r in valid_results]
                
                fig = px.histogram(
                    x=scores,
                    nbins=20,
                    title="Score Distribution",
                    labels={'x': 'Combined Score', 'y': 'Count'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No valid results for score distribution chart")
        
        # Detailed statistics
        st.markdown("#### 📈 Statistical Summary")
        
        if st.session_state.matching_results:
            # Filter out failed results and extract metrics safely
            valid_results = [r for r in st.session_state.matching_results if r.get('score', 0) > 0]
            
            if valid_results:
                all_scores = [r.get('score', 0) for r in valid_results]
                
                # Check pipeline type for appropriate metrics
                pipeline_modes = [r.get('mode', '') for r in valid_results]
                is_two_stage = any(mode.startswith('two_stage') for mode in pipeline_modes)
                is_multi_candidate = any(mode.startswith('multi_candidate') for mode in pipeline_modes)
                
                if is_multi_candidate or is_two_stage:
                    all_dice = [r.get('dice', 0) for r in valid_results]
                    all_ncc = [r.get('ncc', 0) for r in valid_results]
                    all_ssim = [r.get('ssim', 0) for r in valid_results]
                    
                    # Check for rotation information
                    rotation_results = [r for r in valid_results if r.get('best_rotation', 0) != 0]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📊 Final Score", f"{np.mean(all_scores):.4f}", f"±{np.std(all_scores):.4f}")
                        st.metric("🔝 Best Score", f"{np.max(all_scores):.4f}")
                        
                    with col2:
                        st.metric("🎯 Dice Average", f"{np.mean(all_dice):.4f}", f"±{np.std(all_dice):.4f}")
                        st.metric("🎯 Dice Best", f"{np.max(all_dice):.4f}")
                        
                    with col3:
                        st.metric("🔍 NCC Average", f"{np.mean(all_ncc):.4f}", f"±{np.std(all_ncc):.4f}")
                        st.metric("🔍 NCC Best", f"{np.max(all_ncc):.4f}")
                    
                    # # Show multi-candidate specific statistics if available
                    # if is_multi_candidate:
                    #     st.markdown("#### 🏆 Multi-Candidate Analysis")
                        
                    #     # Extract multi-candidate specific metrics
                    #     multi_candidate_results = [r for r in valid_results if r.get('mode', '').startswith('multi_candidate')]
                        
                    #     if multi_candidate_results:
                    #         winning_candidates = [r.get('winning_candidate', 1) for r in multi_candidate_results]
                    #         total_candidates_list = [r.get('total_candidates_processed', 5) for r in multi_candidate_results]
                    #         ranking_changes = [r.get('candidates_ranking_change', 0) for r in multi_candidate_results]
                            
                    #         col1, col2, col3 = st.columns(3)
                            
                    #         with col1:
                    #             avg_winning_rank = np.mean(winning_candidates)
                    #             st.metric("🏆 Average Winning Rank", f"{avg_winning_rank:.1f}")
                                
                    #         with col2:
                    #             avg_candidates_processed = np.mean(total_candidates_list)
                    #             st.metric("📊 Avg Candidates Processed", f"{avg_candidates_processed:.1f}")
                                
                    #         with col3:
                    #             avg_ranking_changes = np.mean(ranking_changes)
                    #             st.metric("🔄 Avg Ranking Changes", f"{avg_ranking_changes:.1f}")
                            
                    #         # Analyze winning candidate distribution
                    #         if winning_candidates:
                    #             st.markdown("##### 📈 Winning Candidate Distribution")
                    #             fig = px.histogram(
                    #                 x=winning_candidates,
                    #                 nbins=max(winning_candidates),
                    #                 title="Which Stage 1 Rank Won in Stage 2?",
                    #                 labels={'x': 'Stage 1 Rank', 'y': 'Times Won'}
                    #             )
                    #             fig.update_layout(height=300)
                    #             st.plotly_chart(fig, use_container_width=True)
                                
                    #             # Insights
                    #             rank_1_wins = winning_candidates.count(1)
                    #             total_multi_results = len(winning_candidates)
                    #             rank_1_percentage = (rank_1_wins / total_multi_results) * 100 if total_multi_results > 0 else 0
                                
                    #             if rank_1_percentage >= 80:
                    #                 st.success(f"🎯 **Excellent Stage 1 Performance**: {rank_1_percentage:.1f}% of Stage 1 top candidates won in Stage 2")
                    #             elif rank_1_percentage >= 60:
                    #                 st.info(f"✅ **Good Stage 1 Performance**: {rank_1_percentage:.1f}% of Stage 1 top candidates won in Stage 2")
                    #             else:
                    #                 st.warning(f"🔄 **Stage 2 Provides Value**: Only {rank_1_percentage:.1f}% of Stage 1 winners remained best after Stage 2 - multi-candidate approach is working!")
                    
                    # Show rotation statistics if rotation testing was used
                    # if rotation_results:
                    #     st.markdown("#### 🔄 Rotation Analysis")
                        
                    #     rotations_applied = [r['best_rotation'] for r in rotation_results]
                    #     rotation_counts = {}
                    #     for rot in rotations_applied:
                    #         rotation_counts[rot] = rotation_counts.get(rot, 0) + 1
                        
                    #     col1, col2, col3 = st.columns(3)
                        
                    #     with col1:
                    #         st.metric("🔄 Rotations Applied", f"{len(rotation_results)}/{len(valid_results)}")
                            
                    #     with col2:
                    #         most_common_rotation = max(rotation_counts.items(), key=lambda x: x[1])
                    #         st.metric("📊 Most Common Rotation", f"{most_common_rotation[0]}°")
                            
                    #     with col3:
                    #         unique_rotations = len(set(rotations_applied))
                    #         st.metric("🎯 Unique Rotations Found", unique_rotations)
                        
                    #     # Show rotation distribution
                    #     if rotations_applied:
                    #         st.markdown("##### 📈 Rotation Distribution")
                    #         fig = px.histogram(
                    #             x=rotations_applied,
                    #             nbins=12,
                    #             title="Applied Rotations Distribution",
                    #             labels={'x': 'Rotation Angle (degrees)', 'y': 'Count'}
                    #         )
                    #         fig.update_layout(height=300)
                    #         st.plotly_chart(fig, use_container_width=True)
                else:
                    # Original single-stage metrics
                    all_ssim = [r.get('ssim', 0) for r in valid_results]
                    all_ncc = [r.get('ncc', 0) for r in valid_results]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📊 Combined Score", f"{np.mean(all_scores):.4f}", f"±{np.std(all_scores):.4f}")
                        st.metric("🔝 Best Score", f"{np.max(all_scores):.4f}")
                        
                    with col2:
                        st.metric("🔍 SSIM Average", f"{np.mean(all_ssim):.4f}", f"±{np.std(all_ssim):.4f}")
                        st.metric("🔍 SSIM Best", f"{np.max(all_ssim):.4f}")
                        
                    with col3:
                        st.metric("🎯 NCC Average", f"{np.mean(all_ncc):.4f}", f"±{np.std(all_ncc):.4f}")
                        st.metric("🎯 NCC Best", f"{np.max(all_ncc):.4f}")
                    
                    with col1:
                        st.metric("📊 Combined Score", f"{np.mean(all_scores):.4f}", f"±{np.std(all_scores):.4f}")
                        st.metric("🔝 Best Score", f"{np.max(all_scores):.4f}")
                        
                    with col2:
                        st.metric("🔍 SSIM Average", f"{np.mean(all_ssim):.4f}", f"±{np.std(all_ssim):.4f}")
                        st.metric("🔍 SSIM Best", f"{np.max(all_ssim):.4f}")
                        
                    with col3:
                        st.metric("🎯 NCC Average", f"{np.mean(all_ncc):.4f}", f"±{np.std(all_ncc):.4f}")
                        st.metric("🎯 NCC Best", f"{np.max(all_ncc):.4f}")
            else:
                st.warning("⚠️ No valid matching results for statistics")
        else:
            st.warning("⚠️ No matching results available for statistics")

with tab5:
    st.header("🔬 3D Visualization")
    st.markdown("Interactive 3D visualization of the best match results with the fossil slice overlay.")
    
    if not st.session_state.matching_done:
        st.info("🔬 Complete the matching process in **3. Run Match** to view 3D visualization")
    elif not st.session_state.best_match_info:
        st.warning("⚠️ No valid matches found for visualization")
    else:
        best_match = st.session_state.best_match_info
        
        # Display best match information
        # st.subheader("🏆 Best Match Information")
        # col1, col2, col3, col4 = st.columns(4)
        # col1.metric("Best Model", best_match['model'].split('_recon')[0].replace("_", " "))
        # col2.metric("Best Score", f"{best_match['score']:.4f}")
        # col3.metric("SSIM", f"{best_match.get('ssim', 0):.4f}")
        # col4.metric("NCC", f"{best_match.get('ncc', 0):.4f}")

        # st.markdown(f"""
        #     <div class="info-card">
        #         <h4>🔬 Visualizing Best Match</h4>
        #         <p><strong>Model:</strong> {best_match['model']}</p>
        #         <p><strong>Species:</strong> {get_species_from_filename(best_match['model'])}</p>
        #         <p><strong>Combined Score:</strong> {best_match['score']:.4f}</p>
        #         <p><strong>Orientation:</strong> {get_orientation_name(best_match.get('axis', 0)).title() if 'axis' in best_match else best_match.get('mode', 'Unknown').title()}</p>
        #         <p><strong>Slice Index:</strong> {best_match.get('index', 'N/A')}</p>
        #     </div>
        # """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="info-card">
                <h4>🔬 Visualizing Best Match</h4>
                <p><strong>Model:</strong> <span style="color: #FFFF00;">{best_match['model']}</span></p>
                <p><strong>Species:</strong> <span style="color: #FFFF00;">{get_species_from_filename(best_match['model'])}</span></p>
                <p>
                    <strong>Combined Score:</strong> <span style="color: #FFFF00;">{best_match['score']:.4f}</span>
                    {f", <strong>Dice:</strong> <span style='color: #FFFF00;'>{best_match.get('dice', 0):.4f}</span>" if best_match.get('mode') in ['two_stage', 'two_stage_rotation_invariant', 'multi_candidate_two_stage', 'multi_candidate_rotation_invariant'] else ""},
                    <strong> SSIM:</strong> <span style="color: #FFFF00;">{best_match.get('ssim', 0):.4f}</span>,
                    <strong> NCC:</strong> <span style="color: #FFFF00;">{best_match.get('ncc', 0):.4f}</span>
                    {f", <strong>ORB:</strong> <span style='color: #FFFF00;'>{best_match.get('orb', 0):.4f}</span>" if 'orb' in best_match else ""}
                </p>
                <p><strong>Orientation:</strong> <span style="color: #FFFF00;">{get_orientation_name(best_match.get('axis', 0)).title() if 'axis' in best_match else best_match.get('mode', 'Unknown').title()}</span></p>
                <p><strong>Slice Index:</strong> <span style="color: #FFFF00;">{best_match.get('index', 'N/A')}</span></p>
            </div>
        """, unsafe_allow_html=True)

                # Extract species and model info
        # model_name = best_match['model'].split('_recon')[0].replace("_", " ")
        # species_name = get_species_from_filename(best_match['model'])
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     st.markdown(f"""
        #         <div class="success-card">
        #             <h4>🦕 {species_name}</h4>
        #             <p><strong>Model:</strong> {model_name}</p>
        #             <p><strong>Orientation:</strong> {get_orientation_name(best_match.get('axis', 0)).title()}</p>
        #             <p><strong>Slice Index:</strong> {best_match.get('index', 'N/A')}</p>
        #         </div>
        #     """, unsafe_allow_html=True)
        
        # with col2:
        #     st.markdown(f"""
        #         <div class="metric-card">
        #             <h4>📊 Match Scores</h4>
        #             <p><strong>Combined Score:</strong> {best_match['score']:.4f}</p>
        #             <p><strong>SSIM:</strong> {best_match.get('ssim', 0):.4f}</p>
        #             <p><strong>NCC:</strong> {best_match.get('ncc', 0):.4f}</p>
        #         </div>
        #     """, unsafe_allow_html=True)
        
        # Visualization controls
        st.subheader("🎛️ Visualization Controls")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_volume = st.checkbox("Show Volume", value=True)
        with col2:
            volume_opacity = st.slider("Volume Opacity", 0.01, 0.2, 0.05, 0.01)
        with col3:
            slice_opacity = st.slider("Slice Opacity", 0.5, 1.0, 0.8, 0.1)
        with col4:
            surface_count = st.slider("Volume Detail", 5, 25, 15, 5)
        
        # Create 3D visualization (simplified for demo)
        if best_match['model'] in st.session_state.loaded_volumes:
            vol_best = st.session_state.loaded_volumes[best_match['model']]
            
            # Downsample for visualization
            vol_ds, (fz, fy, fx) = dynamic_downsample(vol_best)
            Dd, Hd, Wd = vol_ds.shape
            
            # Create meshgrid
            Z, Y, X = np.meshgrid(np.arange(Dd), np.arange(Hd), np.arange(Wd), indexing='ij')
            disp = (vol_ds - vol_ds.min()) / (vol_ds.max() - vol_ds.min() + 1e-8)
            
            # Create 3D plot
            fig = go.Figure()
            
            # Add volume if enabled
            if show_volume:
                fig.add_trace(go.Volume(
                    x=X.ravel(), y=Y.ravel(), z=Z.ravel(),
                    value=disp.ravel(),
                    opacity=volume_opacity,
                    surface_count=surface_count,
                    colorscale="Greys",
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    name="3D Volume"
                ))
            
            # Add slice plane with the matched slice
            slice_added = False
            if 'axis' in best_match and 'index' in best_match:
                slice_idx = best_match['index']
                axis = best_match['axis']
                orientation = get_orientation_name(axis)
                
                # Debug information
                # st.write(f"🔍 Debug: Original slice_idx={slice_idx}, axis={axis}, orientation={orientation}")
                # st.write(f"🔍 Debug: Volume shape (original): {vol_best.shape}")
                # st.write(f"🔍 Debug: Volume shape (downsampled): {vol_ds.shape}")
                # st.write(f"🔍 Debug: Downsampling factors: fz={fz}, fy={fy}, fx={fx}")
                
                try:
                    # Initialize variables
                    slice_data = None
                    x_coords = None
                    y_coords = None
                    z_coords = None
                    
                    # The slice_idx is from the original volume, so we need to map it to the downsampled volume
                    # Since downsampling uses [::fz, ::fy, ::fx], the new index is old_index // downsample_factor
                    if orientation == "axial":
                        # For axial slices (Z-axis), map the original slice index to downsampled volume
                        ds_slice_idx = min(slice_idx // fz, vol_ds.shape[0] - 1)
                        slice_data = vol_ds[ds_slice_idx, :, :]
                        # Create coordinates that match the visualization coordinate system
                        x_coords, y_coords, z_coords = orthogonal_slice_plane_coords(
                            vol_ds.shape, ds_slice_idx, 'axial', 1, 1, 1  # Use unit scaling for consistency
                        )
                    elif orientation == "coronal":
                        # For coronal slices (Y-axis)
                        ds_slice_idx = min(slice_idx // fy, vol_ds.shape[1] - 1)
                        slice_data = vol_ds[:, ds_slice_idx, :]
                        x_coords, y_coords, z_coords = orthogonal_slice_plane_coords(
                            vol_ds.shape, ds_slice_idx, 'coronal', 1, 1, 1
                        )
                    elif orientation == "sagittal":
                        # For sagittal slices (X-axis)
                        ds_slice_idx = min(slice_idx // fx, vol_ds.shape[2] - 1)
                        slice_data = vol_ds[:, :, ds_slice_idx]
                        x_coords, y_coords, z_coords = orthogonal_slice_plane_coords(
                            vol_ds.shape, ds_slice_idx, 'sagittal', 1, 1, 1
                        )
                    else:  # diagonal - use angles if available
                        if 'angles' in best_match:
                            # For diagonal slices, use the angles information
                            angles = best_match['angles']
                            slice_data = vol_ds[vol_ds.shape[0]//2, :, :]  # Default slice for display
                            x_coords, y_coords, z_coords = diagonal_slice_plane_coords(
                                vol_ds.shape, angles, slice_data.shape
                            )
                            ds_slice_idx = vol_ds.shape[0] // 2
                        else:
                            # Fallback to axial
                            ds_slice_idx = vol_ds.shape[0] // 2
                            slice_data = vol_ds[ds_slice_idx, :, :]
                            x_coords, y_coords, z_coords = orthogonal_slice_plane_coords(
                                vol_ds.shape, ds_slice_idx, 'axial', 1, 1, 1
                            )
                    
                    # st.write(f"🔍 Debug: Downsampled slice_idx={ds_slice_idx}")
                    # st.write(f"🔍 Debug: Slice data shape: {slice_data.shape if slice_data is not None else 'None'}")
                    # st.write(f"🔍 Debug: Coordinate shapes: X={x_coords.shape if x_coords is not None else 'None'}, Y={y_coords.shape if y_coords is not None else 'None'}, Z={z_coords.shape if z_coords is not None else 'None'}")
                    
                    # Only proceed if we have valid coordinates and slice data
                    if slice_data is not None and x_coords is not None:
                        # Resize slice data to match coordinate grid if needed
                        target_shape = (x_coords.shape[0], x_coords.shape[1])
                        if slice_data.shape != target_shape:
                            st.write(f"🔍 Debug: Resizing slice from {slice_data.shape} to {target_shape}")
                            try:
                                from skimage.transform import resize
                                slice_data = resize(slice_data, target_shape, anti_aliasing=True, preserve_range=True)
                            except ImportError:
                                # Fallback: use numpy interpolation
                                zoom_y = target_shape[0] / slice_data.shape[0]
                                zoom_x = target_shape[1] / slice_data.shape[1]
                                
                                # Simple nearest neighbor interpolation
                                y_indices = np.round(np.arange(target_shape[0]) / zoom_y).astype(int)
                                x_indices = np.round(np.arange(target_shape[1]) / zoom_x).astype(int)
                                y_indices = np.clip(y_indices, 0, slice_data.shape[0] - 1)
                                x_indices = np.clip(x_indices, 0, slice_data.shape[1] - 1)
                                slice_data = slice_data[np.ix_(y_indices, x_indices)]
                        
                        # Add the slice as a surface
                        fig.add_trace(go.Surface(
                            x=x_coords,
                            y=y_coords,
                            z=z_coords,
                            surfacecolor=slice_data,
                            colorscale='viridis',
                            opacity=slice_opacity,
                            showscale=True,
                            colorbar=dict(title="Slice Intensity", x=1.1),
                            name=f"Best Match Slice ({orientation})"
                        ))
                        
                        slice_added = True
                        st.success(f"✅ Displaying matched {orientation} slice at index {slice_idx} (downsampled: {ds_slice_idx})")
                    
                except Exception as e:
                    st.error(f"❌ Error displaying slice overlay: {str(e)}")
                    import traceback
                    st.error(f"❌ Traceback: {traceback.format_exc()}")
                    
            # Fallback: show a simple plane at the center if no slice was added
            if not slice_added:
                try:
                    center_z = vol_ds.shape[0] // 2
                    slice_data = vol_ds[center_z, :, :]
                    x_coords, y_coords, z_coords = orthogonal_slice_plane_coords(
                        vol_ds.shape, center_z, 'axial', 1, 1, 1
                    )
                    fig.add_trace(go.Surface(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        surfacecolor=slice_data,
                        colorscale='viridis',
                        opacity=slice_opacity,
                        showscale=True,
                        colorbar=dict(title="Slice Intensity", x=1.1),
                        name="Reference Slice (Center)"
                    ))
                    st.info("🔍 Showing reference slice at volume center")
                except Exception as e:
                    st.warning(f"Could not display reference slice: {str(e)}")
            
            # Update layout
            fig.update_layout(
                scene=dict(
                    aspectmode="cube",
                    xaxis=dict(title="X", showgrid=True),
                    yaxis=dict(title="Y", showgrid=True),
                    zaxis=dict(title="Z", showgrid=True),
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                title=f"3D Visualization: {best_match['model']}",
                height=700
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Visualization tips
            st.markdown("""
                **🎛️ Visualization Tips:**
                - 🖱️ Click and drag to rotate the view
                - 🔍 Use mouse wheel to zoom in/out
                - 📊 The visualization shows your matched slice within the 3D model
                - 🎚️ Adjust opacity sliders to better see internal structures
            """)
        else:
            st.error("❌ Volume data not found for visualization")

# Enhanced footer with native Streamlit components
# st.markdown("---")
# st.markdown("## 🦕 AI-Powered Paleontological Analysis Platform")

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.markdown("### 🚀 Technology Stack")
#     st.write("Streamlit • PyTorch • Computer Vision")

# with col2:
#     st.markdown("### 🧠 AI Methods")
#     st.write("SSIM • NCC • Deep Segmentation")

# with col3:
#     st.markdown("### 🔬 Applications")
#     st.write("Species ID • Morphology • Classification")

# st.info("Advanced machine learning algorithms for automated fossil identification and morphological analysis")
