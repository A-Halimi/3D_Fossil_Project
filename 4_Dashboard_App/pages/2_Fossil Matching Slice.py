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
        # Attempt to use a CUDA device
        device = torch.device("cuda:0")
        # Perform a test operation to catch compatibility errors early
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

def apply_fossil_segmentation(slice_np, threshold=0.5, show_preview=False, return_mask=False):
    """
    Enhanced fossil segmentation that focuses purely on fossil content.
    Returns segmented image with background completely removed (set to 0).
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
        # Make threshold more aggressive for better fossil isolation
        global_mean = np.mean(slice_np)
        global_std = np.std(slice_np)
        # Use more aggressive thresholding - higher threshold = more selective
        adaptive_threshold = global_mean + (threshold * 2.0) * global_std  # Doubled sensitivity
        mask_adaptive = slice_np > adaptive_threshold
        
        # Method 3: Percentile-based thresholding (focus on top intensities)
        # Focus on higher percentiles for fossil content
        percentile_threshold = np.percentile(slice_np, 60 + threshold * 35)  # 60-95% range
        mask_percentile = slice_np > percentile_threshold
        
        # Method 4: Histogram-based approach (isolate fossil peaks)
        hist, bins = np.histogram(slice_np.ravel(), bins=50)
        peak_indices = np.where(hist > np.max(hist) * 0.1)[0]  # Find significant peaks
        if len(peak_indices) > 1:
            # Use the higher intensity peak for fossils
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
        if np.sum(mask_combined) < slice_np.size * 0.05:  # Less than 5% of image
            mask_combined = mask_votes >= 2
        
        # Enhanced cleanup for better fossil isolation
        # Remove very small objects (noise)
        if np.sum(mask_combined) > 0:
            mask_cleaned = morphology.remove_small_objects(mask_combined, min_size=200)
        else:
            mask_cleaned = mask_combined
        
        # Fill holes in fossil structures
        mask_filled = ndimage.binary_fill_holes(mask_cleaned)
        
        # Get the largest connected component (main fossil body)
        labeled = measure.label(mask_filled)
        if labeled.max() > 0:
            # Find the largest component
            component_sizes = np.bincount(labeled.flat)[1:]  # Exclude background (0)
            largest_idx = component_sizes.argmax() + 1
            largest_cc = (labeled == largest_idx)
            
            # If the largest component is too small, include more components
            if np.sum(largest_cc) < slice_np.size * 0.02:  # Less than 2%
                # Include top 2-3 largest components
                sorted_indices = np.argsort(component_sizes)[::-1]
                final_mask = np.zeros_like(labeled, dtype=bool)
                for i in range(min(3, len(sorted_indices))):
                    comp_idx = sorted_indices[i] + 1
                    final_mask |= (labeled == comp_idx)
            else:
                final_mask = largest_cc
        else:
            final_mask = mask_filled
        
        # Enhanced morphological operations for cleaner fossil boundaries
        final_mask = morphology.binary_opening(final_mask, morphology.disk(2))
        final_mask = morphology.binary_closing(final_mask, morphology.disk(4))
        
        # Apply the mask to extract pure fossil content
        segmented = original * final_mask
        
        # Enhanced preview with detailed statistics
        if show_preview:
            segmented_display = np.zeros_like(original)
            segmented_display[final_mask] = original[final_mask]
            st.image(segmented_display, caption="✨ Segmented Fossil", use_container_width=True, clamp=True)
                
            # Detailed statistics
            fossil_pixels = np.sum(final_mask)
            total_pixels = final_mask.size
            fossil_percentage = (fossil_pixels / total_pixels) * 100
            
            # Calculate intensity statistics for fossil vs background
            fossil_intensity = np.mean(original[final_mask]) if fossil_pixels > 0 else 0
            background_intensity = np.mean(original[~final_mask]) if np.sum(~final_mask) > 0 else 0
            contrast_ratio = fossil_intensity / (background_intensity + 1e-8)
            
            st.success(f"🎯 **Enhanced Segmentation Results:**\n"
                      f"- Fossil pixels: {fossil_pixels:,} ({fossil_percentage:.1f}% of image)\n"
                      f"- Fossil intensity: {fossil_intensity:.3f}\n"
                      f"- Background intensity: {background_intensity:.3f}\n"
                      f"- Contrast ratio: {contrast_ratio:.2f}x\n"
                      f"- Pure fossil-only matching enabled!")
        
        if return_mask:
            return segmented, final_mask
        return segmented
        
    except ImportError:
        # Fallback if scikit-image is not available
        st.warning("⚠️ Enhanced segmentation requires scikit-image. Using simple thresholding.")
        threshold_val = np.mean(slice_np) + threshold * np.std(slice_np)
        mask = slice_np > threshold_val
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


def segment_volume_slice(volume_slice, threshold=0.5):
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
        
        # Fill holes in fossil structures
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
    if not torch.is_tensor(img1):
        img1 = torch.from_numpy(img1).float()
    if not torch.is_tensor(img2):
        img2 = torch.from_numpy(img2).float()

    if img1.dim() == 2: 
        img1 = img1[None, None]
    if img2.dim() == 2: 
        img2 = img2[None, None]
    img1, img2 = img1.to(img2.dtype), img2

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
    return ssim_map.mean().clamp(-1, 1).item()

def ncc_torch(img1, img2):
    f1, f2 = img1.view(-1), img2.view(-1)
    mean1 = torch.mean(f1); mean2 = torch.mean(f2)
    num = torch.sum((f1-mean1)*(f2-mean2))
    denom = torch.sqrt(torch.sum((f1-mean1)**2)*torch.sum((f2-mean2)**2) + 1e-8)
    return (num/denom).item()

def combined_similarity_torch(ssim_val, ncc_val, w_ssim=0.5, w_ncc=0.5):
    ssim_norm = (ssim_val+1)/2; ncc_norm = (ncc_val+1)/2
    return w_ssim*ssim_norm + w_ncc*ncc_norm

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
                           use_segmentation=False, segmentation_threshold=0.5):
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
                sl = segment_volume_slice(sl, segmentation_threshold)
            
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
                                use_segmentation=False, segmentation_threshold=0.5):
    
    # Search orthogonal slices (with segmentation if enabled)
    best_ortho = search_orthogonal_torch(volume_t, slice_t, w_ssim, w_ncc, axes, fossil_area_threshold,
                                        use_segmentation, segmentation_threshold)
    best = dict(mode="orthogonal", **best_ortho, angles=None, rotation_angle=None, rotation_axis=None)
    
    # Search diagonal slices (optional)
    if enable_diagonal:
        diag = search_diagonal_coarse_to_fine_torch(volume_t, slice_t, w_ssim, w_ncc,
                                                    coarse_step=angle_step,
                                                    angle_list=angle_list) \
               if coarse_to_fine else \
               search_diagonal_brute_force_torch(volume_t, slice_t, w_ssim, w_ncc,
                                                 angle_step, use_all_degrees,
                                                 angle_list)
        if diag["score"] > best["score"]:
            best = dict(mode="diagonal", **diag, axis=None, index=None, rotation_angle=None, rotation_axis=None)
    
    return best

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

def segment_volume_slice(slice_tensor, threshold=0.5):
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
        segmented_np = apply_fossil_segmentation(slice_np, threshold, show_preview=False)
        
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

def ssim_torch(img1, img2, window_size: int = 11, data_range: float = 1.0):
    """Structural SIMilarity for two single‑channel images"""
    if not torch.is_tensor(img1):
        img1 = torch.from_numpy(img1).float()
    if not torch.is_tensor(img2):
        img2 = torch.from_numpy(img2).float()

    if img1.dim() == 2: img1 = img1[None, None]
    if img2.dim() == 2: img2 = img2[None, None]
    img1, img2 = img1.to(img2.dtype), img2

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
    return ssim_map.mean().clamp(-1, 1).item()

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
        if enable_segmentation:
            segmentation_threshold = st.slider("Segmentation Sensitivity", 0.1, 0.9, 0.5, 0.05,
                                             help="Higher = more selective (only brightest fossil parts). Lower = includes more fossil details.")
            
            # Add helpful guidance
            if segmentation_threshold < 0.3:
                st.info("🔍 **Low sensitivity:** Includes more fossil details but may include some background")
            elif segmentation_threshold > 0.7:
                st.info("🎯 **High sensitivity:** Very selective - only brightest fossil structures")
            else:
                st.info("⚖️ **Balanced sensitivity:** Good compromise between detail and background removal")
        
        # Enhanced preprocessing for cropped fossils
        if enable_segmentation:
            # Apply enhanced fossil segmentation
            st.write("🎯 **Applying Enhanced Fossil Segmentation...**")
            
            # Get the segmented result and mask
            segmented_slice, segmentation_mask = apply_fossil_segmentation(slice_np, segmentation_threshold, return_mask=True)
            
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
    
    # Similarity weights configuration
    st.markdown("#### ⚖️ Similarity Weights")
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        w_ssim = st.slider("SSIM Weight", 0.0, 1.0, 0.5, 0.05,
                          help="Weight for Structural Similarity Index (SSIM). Higher values prioritize structural similarity.")
    
    with col_w2:
        w_ncc = 1.0 - w_ssim
        st.metric("NCC Weight", f"{w_ncc:.2f}", 
                 help="Weight for Normalized Cross-Correlation (NCC). Automatically calculated as 1 - SSIM Weight.")
    
    # Add explanation of the weights
    if w_ssim > 0.7:
        st.info("🏗️ **Structure-focused:** High SSIM weight emphasizes structural patterns and shapes")
    elif w_ssim < 0.3:
        st.info("📊 **Intensity-focused:** High NCC weight emphasizes intensity correlations")
    else:
        st.info("⚖️ **Balanced approach:** Equal weighting of structure (SSIM) and intensity (NCC)")
    
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
        
        config_info = f"""
        **🔍 Search Axes:** {axes_options}  
        **🎯 Image Segmentation:** {'Enabled' if segmentation_enabled else 'Disabled'}  
        **↗️ Diagonal Search:** {'Enabled' if enable_diag else 'Disabled'}  
        **⚖️ SSIM Weight:** {w_ssim:.2f}  
        **⚖️ NCC Weight:** {w_ncc:.2f}  
        **🗂️ Models:** {len(selected_models)} selected  
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
                    
                    match_result = match_slice_to_volume_torch(
                        volume_t=vol_t,
                        slice_t=slice_t,
                        w_ssim=w_ssim,
                        w_ncc=w_ncc,
                        axes=axes_options,
                        enable_diagonal=enable_diag,
                        angle_step=30,
                        use_all_degrees=False,
                        coarse_to_fine=True,
                        fossil_area_threshold=0.05,
                        use_segmentation=use_segmentation_for_matching
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
        
        for i, result in enumerate(scores_sorted[:5]):
            rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            species = result.get('species', get_species_from_filename(result['model']))
            
            st.markdown(f"""
                <div class="match-result">
                    <h5>{rank_emoji} {species}</h5>
                    <p><strong>Model:</strong> {result['model']}</p>
                    <p><strong>Combined Score:</strong> {result['score']:.4f}</p>
                    <p><strong>SSIM:</strong> {result['ssim']:.4f} | <strong>NCC:</strong> {result['ncc']:.4f}</p>
                    <p><strong>Mode:</strong> {result['mode']}</p>
                    {f"<p><strong>Axis:</strong> {result['axis']} | <strong>Index:</strong> {result['index']}</p>" if result['mode'] == 'orthogonal' else ""}
                    {f"<p><strong>Angles:</strong> {result['angles']}</p>" if result['mode'] == 'diagonal' else ""}
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
            if best_filtered["mode"] == "orthogonal":
                details = f"Orthogonal axis={best_filtered['axis']}, index={best_filtered['index']}{fossil_info}"
            elif best_filtered["mode"] == "heavy_rotation":
                axis_names = {(0,2): "XZ rotation", (0,1): "XY rotation", (1,2): "YZ rotation"}
                axis_name = axis_names.get(best_filtered.get('rotation_axis'), 'Unknown axis')
                details = f"Heavy Rotation: {best_filtered.get('rotation_angle', 'Unknown')}° {axis_name}, slice {best_filtered.get('slice_index', 'Unknown')}{fossil_info}"
            elif best_filtered["mode"] == "diagonal":
                details = f"Diagonal angles={best_filtered['angles']}"
            else:
                details = f"Mode: {best_filtered['mode']}"
            
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
                    st.metric("SSIM", f"{best_filtered['ssim']:.4f}")
                    st.metric("Combined", f"{best_filtered['score']:.4f}")
                with col2b:
                    st.metric("NCC", f"{best_filtered['ncc']:.4f}")
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
            else:
                st.warning("⚠️ No valid matching results for statistics")
        else:
            st.warning("⚠️ No matching results available for statistics")
            st.metric("🎯 NCC Average", f"{np.mean(all_ncc):.4f}", f"±{np.std(all_ncc):.4f}")
            st.metric("🎯 NCC Best", f"{np.max(all_ncc):.4f}")

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
                    <strong>Combined Score:</strong> <span style="color: #FFFF00;">{best_match['score']:.4f}</span>,
                    <strong> SSIM:</strong> <span style="color: #FFFF00;">{best_match.get('ssim', 0):.4f}</span>,
                    <strong> NCC:</strong> <span style="color: #FFFF00;">{best_match.get('ncc', 0):.4f}</span>
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
