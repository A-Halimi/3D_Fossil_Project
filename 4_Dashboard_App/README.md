# 4_Dashboard_App

## Overview

This section contains an interactive Streamlit web application that provides a user-friendly interface for fossil classification and 3D slice matching using deep learning models and advanced computer vision algorithms. The dashboard features a sophisticated dark theme interface with two main analysis tools for paleontological research.

## Purpose

- Deploy trained ensemble AI models in an interactive web interface
- Provide real-time fossil species classification with confidence analysis
- Enable 3D fossil slice matching using advanced similarity metrics
- Support paleontological research with visual analysis tools
- Demonstrate AI capabilities for fossil identification
- Facilitate comparative analysis between 2D slices and 3D models

## Contents

### Main Application Files

- **[`Home.py`](Home.py)** - Main Streamlit application entry point and welcome page with platform overview
- **[`Run_App.ipynb`](Run_App.ipynb)** - Jupyter notebook for launching the dashboard

### Application Pages

- **[`pages/1_Fossil_DL_Classification.py`](pages/1_Fossil_DL_Classification.py)** - AI-powered fossil species classification interface
- **[`pages/2_Fossil Matching Slice.py`](pages/2_Fossil%20Matching%20Slice.py)** - Advanced 3D fossil slice matching and similarity analysis

### Model Assets

- **[`fossil_classifier_final/`](fossil_classifier_final/)** - Production ensemble model files from Section 3
- **[`models/`](models/)** - 3D fossil models in NIfTI format (194 .nii files) for slice matching

## Features

### 🏠 Home Page (`Home.py`)

The main landing page provides:

#### Welcome Interface
- **Project Overview**: Comprehensive introduction to the Fossil AI Hub platform
- **Platform Capabilities**: Two main analysis tools (AI Classification & 3D Slice Matching)  
- **Navigation Guide**: Step-by-step instructions for using both applications
- **Quick Access**: Navigation buttons to classification and slice matching pages
- **Technology Highlights**: Information about deep learning and computer vision technologies used

#### Key Components
- **Sophisticated Dark Theme**: Custom CSS with gradient backgrounds and animated elements
- **Interactive Cards**: Feature cards with hover effects highlighting platform capabilities
- **Real-time Info**: Current time display (KAUST timezone) and author information
- **Responsive Design**: Mobile-optimized interface with professional styling

### 🦕 AI-Powered Fossil Classification (`pages/1_Fossil_DL_Classification.py`)

Advanced deep learning interface for fossil species identification:

#### Core Classification Features
- **Ensemble Model**: ConvNeXt-Large + EfficientNet-V2 architecture for robust predictions
- **12 Species Support**: Classification of benthic foraminifera species
- **Real-time Processing**: Instant image analysis with sophisticated preprocessing
- **Interactive Segmentation**: Live preview with adjustable threshold sensitivity slider

#### Image Processing Pipeline
```python
# Processing methods available:
- Auto-Process: Automatic fossil detection and segmentation
- Manual Crop: User-defined region selection with bounding box tool
- Threshold Adjustment: Fine-tunable segmentation sensitivity (-50 to +50)
- Live Preview: Real-time visualization of processed image fed to AI model
```

#### Advanced Analysis Dashboard
- **Confidence Metrics**: Detailed probability distribution across all species
- **Top Predictions**: Ranked classification results with confidence percentages
- **Statistical Analysis**: Entropy, cumulative confidence, and distribution metrics
- **Interactive Visualizations**: 
  - Full species probability distribution with color-coded confidence levels
  - Top 10 predictions with ranking visualization
  - Confidence distribution histogram with interpretation guides

#### Technical Specifications
- **Input Size**: 384×384 pixels with automatic padding and cropping
- **Model Architecture**: PatchEnsemble with weak learner detection
- **Preprocessing**: Adaptive segmentation with structure preservation
- **Output Format**: Probability distributions with detailed statistical analysis

### 🎯 3D Fossil Slice Matching (`pages/2_Fossil Matching Slice.py`)

Sophisticated 3D correspondence analysis using multiple similarity metrics:

#### Core Matching Technology
- **Multi-Metric Similarity**: SSIM (Structural Similarity Index), NCC (Normalized Cross-Correlation), and Dice Score
- **3D Model Database**: 97 NIfTI (.nii) format 3D fossil models from multiple species
- **Rotation Invariance**: Testing across multiple rotation angles (20°, 40°, 60°, 80°, 100°, 120°, 140°, 160°)
- **ORB Feature Matching**: Additional feature-based similarity analysis

#### Advanced Processing Pipeline
```python
# Two-stage matching process:
Stage 1: Multi-Candidate Coarse Search
- Dice coefficient and Hu Moments analysis
- Cross-sectional slice extraction from 3D models
- Fossil content detection and validation

Stage 2: Fine-tuning with Rotation Analysis  
- SSIM and NCC metric computation
- Rotation testing for orientation invariance
- ORB feature point matching
- Combined scoring with weighted metrics
```

#### Application Interface Structure
**Tab 1: Upload & Setup**
- Enhanced file uploader with drag-and-drop support
- Automatic image preprocessing and segmentation
- Structure-preserving fossil extraction
- Real-time processing feedback

**Tab 2: Model Selection**  
- Built-in 3D model library browsing (97 models)
- Species-based filtering and selection
- Custom model upload capability (.nii/.nii.gz format)
- Model metadata display and statistics

**Tab 3: Run Analysis**
- Configurable similarity metrics weighting
- Batch processing across selected models  
- Real-time progress tracking with detailed logging
- Multi-threaded analysis for performance

**Tab 4: Results & Stats**
- Comprehensive matching results with ranked similarity scores
- Side-by-side visual comparison of 2D slice vs. 3D model slices
- Statistical analysis of matching confidence
- Species identification based on best matches

**Tab 5: 3D Visualization**
- Interactive 3D model exploration
- Slice-by-slice correspondence visualization
- Rotation and orientation analysis display
- Match highlighting and annotation

## Getting Started

### Prerequisites

Ensure you have the required environment set up with the necessary dependencies. The application requires:

- **Python 3.8+** with TensorFlow 2.13+
- **Streamlit 1.25+** for the web interface
- **PyTorch** for 3D similarity computations
- **nibabel** for NIfTI medical imaging format support
- **OpenCV, scikit-image** for image processing
- **Plotly** for interactive visualizations

### Installation and Setup

### Recommended Environment

```bash
# Use the provided Docker container for best compatibility
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --rm -p 10000:8888 -p 8501:8501 -v ${PWD}:/workspace/mycode \
  abdelghafour1/ngc_tf_rapids_25_01_vscode_torch:2025-v3 \
  jupyter lab --ip=0.0.0.0 --allow-root \
  --NotebookApp.custom_display_url=http://$(hostname):10000
```

#### 1. Verify Model Files

```bash
cd 4_Dashboard_App

# Ensure the ensemble AI model is available
ls fossil_classifier_final/
# Should contain: model.keras, class_names.json

# Verify 3D fossil models are present
ls models/
# Should contain: 194 .nii files (3D fossil models)
```

#### 2. Install Required Dependencies

```bash
# Install core Streamlit dependencies
pip install streamlit plotly seaborn pillow pandas

# Install deep learning libraries  
pip install tensorflow torch torchvision

# Install scientific computing and image processing
pip install opencv-python scikit-image nibabel scipy numpy

# Optional: Enhanced UI components
pip install streamlit-extras
```

#### 3. Launch the Dashboard

**Method 1: Using Streamlit directly**

```bash
cd 4_Dashboard_App
streamlit run Home.py --server.maxUploadSize 9000 --server.maxMessageSize 10000
```

**Method 2: Using the Jupyter notebook**

```bash
# Open and run the notebook
jupyter lab Run_App.ipynb
```

#### 4. Access the Dashboard

Once launched, access the dashboard at:
- **Local**: `http://localhost:8501`
- **Network**: `http://[your-ip]:8501`

### Application Workflow

#### For AI Classification:
1. **Navigate** to "Fossil DL Classification" page via sidebar
2. **Upload** your fossil image (JPG, PNG, TIFF formats)
3. **Choose** processing method (Auto-Process or Manual Crop)
4. **Adjust** segmentation threshold using the sensitivity slider
5. **Preview** the processed image in real-time
6. **Classify** and explore detailed probability analysis

#### For 3D Slice Matching:
1. **Navigate** to "3D Fossil Slice Matcher" page via sidebar  
2. **Upload** your 2D fossil slice image
3. **Select** 3D models from the built-in database (97 models available)
4. **Configure** similarity metrics (SSIM, NCC, Dice weights)
5. **Run** comprehensive matching analysis
6. **Explore** results with 3D visualization and statistical analysis


## Advanced Features

### Real-time Image Processing

The classification system implements sophisticated preprocessing:

```python
# Live segmentation with adjustable sensitivity
def segment_image(img_gray, threshold_adjustment=0.0):
    """
    Structure-preserving fossil segmentation with user control
    - Otsu thresholding with manual adjustment (-50 to +50)
    - Morphological operations for noise reduction
    - Connected component analysis for main fossil isolation
    - Real-time preview generation for user feedback
    """
    
# Automatic bounding box detection and cropping
@tf.autograph.experimental.do_not_convert  
def crop_to_bbox(img):
    """
    TensorFlow-optimized automatic fossil detection
    - Edge detection and coordinate extraction
    - Dynamic padding and resizing to canonical size
    - Handles edge cases and empty detections
    """
```

### Multi-Metric 3D Similarity Analysis

```python
# Advanced similarity computation pipeline
class SimilarityAnalyzer:
    def compute_comprehensive_similarity(self, slice_2d, model_3d):
        """
        Multi-stage analysis pipeline:
        1. Cross-sectional slice extraction from 3D model
        2. Structural Similarity Index (SSIM) computation
        3. Normalized Cross-Correlation (NCC) analysis  
        4. Dice coefficient for overlap measurement
        5. ORB feature matching for geometric correspondence
        6. Rotation invariance testing across 8 angles
        7. Combined weighted scoring for final ranking
        """
```

### Interactive Visualization System

The dashboard provides sophisticated data visualization capabilities:

```python
# Plotly-based interactive charts with dark theme optimization
def create_probability_visualization(predictions, class_names):
    """
    Multi-tab visualization system:
    - Full distribution bar chart with color-coded confidence levels
    - Top 10 predictions with ranking and medal system  
    - Confidence histogram with statistical interpretation
    - Real-time updating with smooth transitions
    """
    
# 3D model visualization and slice correspondence
def visualize_3d_correspondence(model_data, matching_results):
    """
    Advanced 3D visualization features:
    - Interactive 3D fossil model rendering
    - Slice-by-slice correspondence highlighting
    - Rotation analysis visualization
    - Match confidence overlay mapping
    """
```

## Performance Optimization

### Caching and Memory Management

```python
# Strategic caching for improved performance
@st.cache_resource(show_spinner="Loading AI model...")
def load_ensemble_model():
    """Cache TensorFlow model loading with optimized memory usage"""
    
@st.cache_data
def load_3d_model_database():
    """Cache 3D model metadata for faster browsing"""

# Automatic memory management
def optimize_memory_usage():
    """
    - Periodic TensorFlow session clearing
    - Garbage collection for large 3D volumes
    - Session state cleanup for performance
    """
```

### GPU Acceleration

```python
# Optimized device configuration with fallbacks
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Mixed precision for faster inference
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except RuntimeError:
    # Graceful fallback to CPU processing
    device = torch.device("cpu")
```

## Technical Specifications

### System Requirements

- **Python**: 3.8+ with scientific computing stack
- **Deep Learning**: TensorFlow 2.13+, PyTorch 1.12+
- **Web Framework**: Streamlit 1.25+ with enhanced components
- **Image Processing**: OpenCV 4.5+, scikit-image 0.19+
- **3D Analysis**: nibabel for NIfTI format support
- **Memory**: 16GB+ RAM recommended (32GB+ for large 3D models)
- **GPU**: Optional but recommended for faster inference

### Browser Compatibility

- **Chrome**: 90+ (recommended for best performance)
- **Firefox**: 85+ (full feature support)
- **Safari**: 14+ (WebGL required for 3D visualization)
- **Edge**: 90+ (Chromium-based versions)


## Connection to Other Sections

### Input Dependencies
- **From Section 3**: Production ensemble model (`fossil_classifier_final/`)
- **From Section 1**: 3D NIfTI fossil models (`models/` directory with 194 files)
- **Model Architecture**: ConvNeXt-Large + EfficientNet-V2 ensemble
- **Species Coverage**: 12 benthic foraminifera classes

### Integration Points
- **Seamless Model Loading**: Direct integration with trained TensorFlow models
- **Preprocessing Consistency**: Same pipeline as training (crop_to_bbox, segmentation)
- **Real-time Analysis**: Live feedback and interactive parameter adjustment
- **3D Correspondence**: Advanced similarity metrics for slice-to-model matching

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Verify `fossil_classifier_final/model.keras` exists
2. **3D Model Access**: Ensure `models/` directory contains .nii files (97 total)
3. **Memory Issues**: Reduce batch size or restart application periodically
4. **Slow Performance**: Check GPU availability and optimize memory settings

### Performance Issues

1. **Large Image Processing**: Application automatically handles image resizing
2. **3D Model Loading**: Models are loaded on-demand to optimize memory
3. **Concurrent Users**: Designed for single-user analysis sessions
4. **Browser Memory**: Clear browser cache if visualization becomes sluggish

This dashboard represents the culmination of the entire fossil classification pipeline, providing an accessible and sophisticated interface for utilizing advanced AI models and 3D analysis techniques in paleontological research.
