# =========================================================================
#  Fossil AI Hub - HOME PAGE
#  Advanced AI-Powered Paleontological Analysis Platform
# =========================================================================

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fossil AI Hub | Home", 
    page_icon="🏠", 
    initial_sidebar_state="expanded",
    layout="wide"
)

# Custom CSS function matching the sophisticated design
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
    
    .feature-card {
        background: linear-gradient(135deg, var(--background-tertiary) 0%, var(--background-secondary) 100%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        color: var(--text-primary);
        box-shadow: var(--shadow-medium);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: var(--shadow-large);
        border-color: var(--accent-color);
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color), var(--accent-color));
        border-radius: 16px 16px 0 0;
    }
    
    /* Icon enhancements */
    .icon {
        display: inline-block;
        margin-right: 0.5rem;
        font-size: 1.2em;
    }
    
    .large-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    /* Navigation button styling */
    .nav-button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        color: white;
        font-weight: 600;
        font-size: 1rem;
        text-decoration: none;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: var(--shadow-soft);
        transition: all 0.3s ease;
    }
    
    .nav-button:hover {
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
        box-shadow: var(--shadow-medium);
        transform: translateY(-2px);
        text-decoration: none;
        color: white;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .feature-card {
            margin: 0.5rem 0;
            padding: 1.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Apply the CSS
inject_custom_css()

# ---------------------------------------------------------------------------
# 2. SIDEBAR LAYOUT
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;">
            <h2 style="margin: 0; color: #ffffff; font-size: 1.5rem;">🏠 Fossil AI Hub</h2>
            <p style="margin: 0.5rem 0 0 0; color: #c7d2fe; font-size: 0.9rem;">Advanced Paleontological Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="height: 1px; background: linear-gradient(90deg, transparent, #374151, transparent); margin: 1rem 0;"></div>
    """, unsafe_allow_html=True)
    
    # Navigation section
    st.markdown("""
        <div class="glass-card" style="padding: 1rem; margin-bottom: 1.5rem; text-align: center;">
            <h4 style="margin: 0 0 1rem 0; color: #ffffff; display: flex; align-items: center; justify-content: center;">
                <span class="icon">🧭</span>Navigation
            </h4>
            <p style="color: #c7d2fe; font-size: 0.9rem; margin: 0;">
                Select a page from the sidebar menu to begin your fossil analysis journey
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Platform features
    st.markdown("""
        <div class="info-card">
            <h5 style="margin: 0 0 0.75rem 0; color: white; font-weight: 600;">
                <span class="icon">🚀</span>Platform Features
            </h5>
            <ul style="margin: 0; padding-left: 1.2rem; color: rgba(255,255,255,0.9); font-size: 0.85rem; line-height: 1.4;">
                <li><strong>AI Classification:</strong> Deep learning species identification</li>
                <li><strong>3D Matching:</strong> Advanced slice-to-model correspondence</li>
                <li><strong>Interactive UI:</strong> Real-time segmentation preview</li>
                <li><strong>Advanced Analytics:</strong> Comprehensive statistical insights</li>
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
# 3. MAIN PAGE CONTENT
# ---------------------------------------------------------------------------

# Main header
st.markdown("""
    <div class="main-header">
        <div style="position: relative; z-index: 2;">
            <h1>🦕 Fossil AI Hub</h1>
            <p style="font-size: 1.4rem; margin-bottom: 0.5rem;">Harnessing Advanced AI to Decode Fossil Rock Slice Images for Species Identification</p>
            <p style="font-size: 1.1rem; opacity: 0.85;">
                Deep learning and 3D matching technology for paleontological research
            </p>
            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 12px; backdrop-filter: blur(10px);">
                <p style="margin: 0; font-size: 0.95rem;">
                    <span style="color: #00D4FF;">🧠 AI Technology:</span> <strong style="color: #FFFFFF;">Deep Learning + Computer Vision</strong> |
                    <span style="color: #FF6B35;">🔬 Analysis:</span> <strong style="color: #FFFFFF;">Species Classification + 3D Matching</strong> |
                    <span style="color: #FFE66D;">🎯 Precision:</span> <strong style="color: #FFFFFF;">Advanced Ensemble Models</strong>
                </p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Welcome section
st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h2 style="color: #ffffff; margin-bottom: 1rem;">
            <span class="icon">👋</span>Welcome to the Fossil Classifier App!
        </h2>
        <p style="color: #c7d2fe; font-size: 1.2rem; margin: 0; max-width: 800px; margin: 0 auto;">
            This application harnesses state-of-the-art AI to revolutionize paleontological analysis through advanced fossil identification and 3D matching capabilities.
        </p>
    </div>
""", unsafe_allow_html=True)

# Features section
st.markdown("""
    <div style="margin: 3rem 0;">
        <h2 style="color: #ffffff; margin-bottom: 2rem; text-align: center;">
            <span class="icon">✨</span>Platform Capabilities
        </h2>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="feature-card">
            <div style="text-align: center;">
                <span class="large-icon">🤖</span>
                <h3 style="margin: 0 0 1rem 0; color: #00D4FF;">AI-Powered Classification</h3>
                <p style="color: #c7d2fe; line-height: 1.6; margin-bottom: 1.5rem;">
                    Classify fossil images using our advanced deep learning model powered by ConvNeXt and EfficientNet ensemble architecture.
                </p>
                <ul style="color: rgba(255,255,255,0.9); font-size: 0.9rem; line-height: 1.5; text-align: left;">
                    <li><strong>12 Species:</strong> Benthic foraminifera classification</li>
                    <li><strong>Ensemble AI:</strong> Dual-model architecture</li>
                    <li><strong>Real-time:</strong> Interactive segmentation preview</li>
                    <li><strong>Confidence:</strong> Detailed probability analysis</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="feature-card">
            <div style="text-align: center;">
                <span class="large-icon">🎯</span>
                <h3 style="margin: 0 0 1rem 0; color: #FF6B35;">3D Slice Matching</h3>
                <p style="color: #c7d2fe; line-height: 1.6; margin-bottom: 1.5rem;">
                    Match 2D fossil slices with 3D fossil models using advanced similarity metrics including SSIM and NCC algorithms.
                </p>
                <ul style="color: rgba(255,255,255,0.9); font-size: 0.9rem; line-height: 1.5; text-align: left;">
                    <li><strong>SSIM + NCC:</strong> Advanced similarity metrics</li>
                    <li><strong>3D Visualization:</strong> Interactive model exploration</li>
                    <li><strong>Slice Matching:</strong> Precise correspondence mapping</li>
                    <li><strong>Multi-angle:</strong> Comprehensive 3D analysis</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

# How to use section
st.markdown("""
    <div style="margin: 3rem 0;">
        <h2 style="color: #ffffff; margin-bottom: 2rem; text-align: center;">
            <span class="icon">📚</span>How to Use the Platform
        </h2>
    </div>
""", unsafe_allow_html=True)

# Step-by-step guide
steps_col1, steps_col2 = st.columns(2)

with steps_col1:
    st.markdown("""
        <div class="success-card">
            <h4 style="margin: 0 0 1rem 0; color: white; text-align: center;">
                <span class="icon">📊</span>Deep Learning Classification
            </h4>
            <ol style="color: rgba(255,255,255,0.9); font-size: 0.95rem; line-height: 1.6; padding-left: 1.2rem;">
                <li><strong>Upload Image:</strong> Select your fossil image (JPG, PNG, TIFF)</li>
                <li><strong>Choose Method:</strong> Auto-process or manual crop</li>
                <li><strong>Adjust Settings:</strong> Fine-tune segmentation threshold</li>
                <li><strong>Classify:</strong> Get AI-powered species predictions</li>
                <li><strong>Analyze Results:</strong> Explore confidence metrics and charts</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

with steps_col2:
    st.markdown("""
        <div class="info-card">
            <h4 style="margin: 0 0 1rem 0; color: white; text-align: center;">
                <span class="icon">🎯</span>3D Slice Matching
            </h4>
            <ol style="color: rgba(255,255,255,0.9); font-size: 0.95rem; line-height: 1.6; padding-left: 1.2rem;">
                <li><strong>Upload Slice:</strong> Select your 2D fossil slice image</li>
                <li><strong>Choose Model:</strong> Select 3D fossil model for matching</li>
                <li><strong>Set Parameters:</strong> Configure similarity algorithms</li>
                <li><strong>Run Analysis:</strong> Execute SSIM + NCC matching</li>
                <li><strong>Visualize:</strong> Explore 3D correspondence results</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# Navigation buttons section
st.markdown("""
    <div style="margin: 3rem 0;">
        <h2 style="color: #ffffff; margin-bottom: 2rem; text-align: center;">
            <span class="icon">🚀</span>Quick Navigation
        </h2>
    </div>
""", unsafe_allow_html=True)

# Quick access buttons
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("🤖 Deep Learning Classifier", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Fossil_DL_Classification.py")

with nav_col2:
    if st.button("🎯 3D Slice Matcher", use_container_width=True):
        st.switch_page("pages/2_Fossil Matching Slice.py")

with nav_col3:
    if st.button("📖 Documentation", use_container_width=True):
        st.info("Documentation coming soon!")

# Instructions section
st.markdown("""
    <div style="margin: 3rem 0;">
        <div class="glass-card" style="padding: 2rem;">
            <h3 style="margin: 0 0 1.5rem 0; color: #ffffff; text-align: center;">
                <span class="icon">📋</span>Getting Started Instructions
            </h3>
            <div style="color: #c7d2fe; font-size: 1rem; line-height: 1.7;">
                <p style="margin-bottom: 1rem;">
                    <strong>🔍 Navigation:</strong> Select a page from the sidebar to begin, or click one of the navigation buttons above if they work in your deployment.
                </p>
                <p style="margin-bottom: 1rem;">
                    <strong>📊 For Classification:</strong> Upload an image on the <em>Deep Learning</em> page to see the predicted fossil species with confidence metrics and detailed analysis.
                </p>
                <p style="margin-bottom: 1rem;">
                    <strong>🎯 For Matching:</strong> Match a 2D slice on the <em>Slice Matcher</em> page to visualize the 3D correspondence using advanced similarity algorithms.
                </p>
                <p style="margin: 0;">
                    <strong>💡 Pro Tip:</strong> Both applications feature interactive controls for optimal results - experiment with different settings to achieve the best analysis for your specific fossil samples.
                </p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer section
st.markdown("""
    <div style="margin-top: 4rem; padding-top: 2rem; border-top: 1px solid #374151;">
        <div style="text-align: center;">
            <p style="color: #9ca3af; font-size: 0.9rem; margin: 0;">
                🦕 Fossil AI Hub | Advanced Paleontological Analysis Platform | Powered by Deep Learning & Computer Vision
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)
