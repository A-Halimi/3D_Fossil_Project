# 4_Dashboard_App

## Overview

This section contains an interactive Streamlit web application that provides a user-friendly interface for fossil classification using the trained deep learning models from previous sections. The dashboard allows users to upload fossil images, get real-time predictions, explore the dataset, and visualize model performance.

## Purpose

- Deploy trained models in an interactive web interface
- Provide real-time fossil species classification
- Enable dataset exploration and visualization
- Demonstrate model capabilities to end users
- Support research and educational activities
- Facilitate model testing and validation

## Contents

### Main Application Files

- **[`Home.py`](Home.py)** - Main Streamlit application entry point and home page
- **[`Run_App.ipynb`](Run_App.ipynb)** - Jupyter notebook for launching the dashboard

### Application Pages

- **[`pages/1_Fossil_DL_Classification.py`](pages/1_Fossil_DL_Classification.py)** - Main classification interface
- **[`pages/2_Fossil Matching Slice.py`](pages/2_Fossil%20Matching%20Slice.py)** - Fossil slice matching and comparison tools

### Model Assets

- **[`fossil_classifier_final/`](fossil_classifier_final/)** - Production model files copied from Section 3
- **[`models/`](models/)** - Additional model artifacts and utilities

## Features

### 🔬 Fossil Classification Interface

#### Real-time Image Classification
- **Drag & Drop Upload**: Easy image upload interface
- **Instant Predictions**: Real-time species classification
- **Confidence Scores**: Probability distribution across all species
- **Multiple Format Support**: PNG, JPG, JPEG image formats

#### Advanced Analysis Tools
- **Prediction Confidence**: Visual confidence indicators
- **Top-N Predictions**: Shows top 3 most likely species
- **Model Comparison**: Compare predictions across different models
- **Batch Processing**: Classify multiple images simultaneously

### 📊 Dataset Exploration

#### Interactive Data Visualization
- **Species Distribution**: Visual breakdown of dataset composition
- **Sample Gallery**: Browse representative samples from each species
- **Dataset Statistics**: Comprehensive dataset metrics
- **Quality Metrics**: Image quality and preprocessing information

#### Educational Content
- **Species Information**: Detailed descriptions of each fossil species
- **Morphological Features**: Key identifying characteristics
- **Historical Context**: Geological and paleontological background

### 🎯 Model Performance Dashboard

#### Performance Metrics
- **Accuracy Metrics**: Overall and per-class performance
- **Confusion Matrix**: Interactive confusion matrix visualization
- **ROC Curves**: Receiver Operating Characteristic analysis
- **Training History**: Model training progress visualization

#### Model Comparison Tools
- **Architecture Comparison**: Side-by-side model performance
- **Inference Speed**: Timing benchmarks for different models
- **Resource Usage**: Memory and computational requirements

## Getting Started

### Prerequisites

Ensure you're in the recommended Docker environment and have completed Sections 1-3:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -p 10000:8888 -p 8501:8501 -v ${PWD}:/workspace/mycode abdelghafour1/ngc_tf_rapids_25_01_vscode_torch:2025-v3 jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://$(hostname):10000
```

### Installation and Setup

#### 1. Verify Model Files

```bash
cd 4_Dashboard_App

# Ensure the production model is available
ls fossil_classifier_final/
# Should contain: model.keras, class_names.json, etc.

# If missing, copy from Section 3 results
cp -r ../3_Results/fossil_classifier_final/* fossil_classifier_final/
```

#### 2. Install Required Dependencies

```bash
# Install Streamlit and dependencies (if not already installed)
pip install streamlit plotly seaborn pillow
```

#### 3. Launch the Dashboard

**Method 1: Using Streamlit directly**

```bash
cd 4_Dashboard_App
streamlit run Home.py --server.port 8501 --server.address 0.0.0.0
```

**Method 2: Using the Jupyter notebook**

```bash
# Open and run the notebook
jupyter lab Run_App.ipynb
```

#### 4. Access the Dashboard

Once launched, access the dashboard at:
- **Local**: `http://localhost:8501`
- **Docker**: `http://$(hostname):8501`
- **Network**: `http://[your-ip]:8501`

## Application Structure

### Home Page (`Home.py`)

The main landing page provides:

```python
# Key features of Home.py:
- Project overview and introduction
- Navigation to different app sections
- Quick start guide for new users
- Model performance summary
- Dataset statistics overview
```

#### Welcome Interface
- **Project Description**: Overview of the fossil classification project
- **Model Information**: Details about the ensemble model used
- **Usage Instructions**: Step-by-step guide for new users
- **Performance Highlights**: Key accuracy metrics and achievements

### Classification Page (`pages/1_Fossil_DL_Classification.py`)

Main classification interface with:

#### Upload Interface
```python
# File upload component
uploaded_file = st.file_uploader(
    "Choose a fossil image...",
    type=['png', 'jpg', 'jpeg'],
    help="Upload a fossil image for classification"
)
```

#### Prediction Display
```python
# Prediction results visualization
if prediction_made:
    st.subheader("🔍 Classification Results")
    
    # Top prediction
    st.metric(
        label="Predicted Species",
        value=predicted_species,
        delta=f"{confidence:.1%} confidence"
    )
    
    # Probability distribution
    plot_prediction_probabilities(predictions, class_names)
    
    # Additional insights
    display_species_information(predicted_species)
```

#### Advanced Features
- **Image Preprocessing Visualization**: Show preprocessing steps
- **Attention Maps**: Highlight important image regions (if available)
- **Similar Samples**: Find similar fossils in the training dataset
- **Export Results**: Download prediction results

### Fossil Matching Page (`pages/2_Fossil Matching Slice.py`)

Specialized tool for comparing fossil slices:

#### Slice Comparison
- **Side-by-side Display**: Compare multiple fossil images
- **Similarity Scoring**: Quantitative similarity measures
- **Feature Matching**: Highlight matching morphological features
- **Database Search**: Find similar specimens in the dataset

#### Analysis Tools
- **Morphometric Analysis**: Automated shape and size measurements
- **Texture Analysis**: Surface pattern comparison
- **3D Reconstruction Preview**: Link to original 3D data
- **Expert Annotations**: Add and view expert comments

## Configuration and Customization

### Model Configuration

```python
# config/model_config.py
MODEL_CONFIG = {
    "model_path": "fossil_classifier_final/model.keras",
    "class_names_path": "fossil_classifier_final/class_names.json",
    "preprocessing": {
        "target_size": (224, 224),
        "rescale": 1.0/255.0,
        "color_mode": "rgb"
    },
    "prediction": {
        "batch_size": 1,
        "top_k": 3,
        "confidence_threshold": 0.5
    }
}
```

### UI Customization

```python
# config/ui_config.py
UI_CONFIG = {
    "page_title": "Fossil Classification Dashboard",
    "page_icon": "🦴",
    "layout": "wide",
    "theme": {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6"
    },
    "sidebar": {
        "show_species_info": True,
        "show_model_metrics": True,
        "show_dataset_stats": True
    }
}
```

### Species Information Database

```python
# data/species_info.json
{
    "Alveolina": {
        "description": "Large benthic foraminifera...",
        "geological_age": "Paleocene to Eocene",
        "morphology": "Elongated, fusiform shape...",
        "distribution": "Tropical shallow marine environments",
        "identifying_features": [
            "Complex internal structure",
            "Spiral arrangement of chambers",
            "Thick walls"
        ]
    },
    // ... other species
}
```

## Advanced Features

### Model Ensemble Integration

The dashboard supports multiple model backends:

```python
class ModelEnsemble:
    def __init__(self):
        self.main_model = load_model("fossil_classifier_final/model.keras")
        self.backup_models = {
            "convnext": load_model("models/convnext_best.keras"),
            "effv2s": load_model("models/effv2s_best.keras")
        }
    
    def predict_ensemble(self, image):
        # Weighted ensemble prediction
        predictions = []
        weights = [0.6, 0.25, 0.15]  # Main, ConvNeXt, EfficientNet
        
        for model, weight in zip(self.models, weights):
            pred = model.predict(image)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)
```

### Real-time Performance Monitoring

```python
def track_prediction_metrics():
    """Monitor dashboard usage and model performance"""
    metrics = {
        "total_predictions": st.session_state.get('prediction_count', 0),
        "average_confidence": st.session_state.get('avg_confidence', 0),
        "most_predicted_species": st.session_state.get('top_species', []),
        "processing_time": st.session_state.get('avg_time', 0)
    }
    return metrics
```

### Batch Processing Interface

```python
def batch_classification():
    """Process multiple images simultaneously"""
    uploaded_files = st.file_uploader(
        "Upload multiple fossil images",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Classify All"):
        results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            result = classify_single_image(file)
            results.append(result)
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        display_batch_results(results)
```

## Performance Optimization

### Caching Strategies

```python
@st.cache_resource
def load_model():
    """Cache model loading for faster startup"""
    return tf.keras.models.load_model("fossil_classifier_final/model.keras")

@st.cache_data
def load_species_info():
    """Cache species information database"""
    with open("data/species_info.json", 'r') as f:
        return json.load(f)

@st.cache_data
def preprocess_image(image_data):
    """Cache image preprocessing results"""
    # Preprocessing logic
    return processed_image
```

### Memory Management

```python
def optimize_memory_usage():
    """Implement memory optimization strategies"""
    # Clear TensorFlow session periodically
    if st.session_state.get('prediction_count', 0) % 100 == 0:
        tf.keras.backend.clear_session()
    
    # Garbage collection for large images
    import gc
    gc.collect()
```

## Deployment Options

### Local Development

```bash
# Development mode with auto-reload
streamlit run Home.py --server.runOnSave true
```

### Production Deployment

```bash
# Production mode with optimizations
streamlit run Home.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.maxUploadSize 50 \
    --server.enableCORS false \
    --server.enableXsrfProtection true
```

### Docker Deployment

```dockerfile
# Dockerfile for standalone deployment
FROM tensorflow/tensorflow:2.13.0-gpu

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY . /app/
WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Connection to Other Sections

### Input Dependencies
- **From Section 3**: Production model (`fossil_classifier_final/`)
- **Model Artifacts**: Trained models and evaluation metrics
- **Dataset Information**: Species data and sample images

### Integration Points
- **Model Loading**: Seamless integration with TensorFlow models
- **Preprocessing Pipeline**: Consistent with training preprocessing
- **Performance Metrics**: Real-time model performance tracking

## User Guide

### For Researchers

1. **Upload Test Images**: Use the classification interface for quick species identification
2. **Batch Analysis**: Process multiple specimens simultaneously
3. **Export Results**: Download predictions for further analysis
4. **Compare Models**: Evaluate different model architectures

### For Educators

1. **Interactive Learning**: Explore fossil diversity through the interface
2. **Visual Examples**: Use the gallery for teaching morphological features
3. **Real-time Demos**: Demonstrate AI classification in classroom settings
4. **Species Information**: Access detailed paleontological information

### For Students

1. **Hands-on Experience**: Practice fossil identification with AI assistance
2. **Learn by Comparison**: Compare your identifications with AI predictions
3. **Explore Dataset**: Browse the comprehensive fossil image collection
4. **Understand AI**: Learn how deep learning works in paleontology

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Verify model files are in correct locations
2. **Image Upload Issues**: Check file formats and size limits
3. **Slow Predictions**: Monitor GPU/CPU usage and memory
4. **UI Rendering Issues**: Clear browser cache and restart Streamlit

### Performance Issues

1. **Memory Leaks**: Restart app periodically for long sessions
2. **Slow Loading**: Implement additional caching strategies
3. **Concurrent Users**: Consider scaling solutions for multiple users
4. **Large Images**: Implement client-side image compression

## Future Enhancements

### Planned Features

1. **3D Visualization**: Integration with original 3D fossil data
2. **Uncertainty Quantification**: Advanced confidence estimation
3. **Active Learning**: Collect user feedback for model improvement
4. **Mobile Optimization**: Enhanced mobile interface design
5. **API Endpoints**: REST API for programmatic access

### Advanced Analytics

1. **Usage Analytics**: Track user behavior and popular features
2. **Model Drift Detection**: Monitor model performance over time
3. **A/B Testing**: Compare different model versions
4. **User Feedback Integration**: Continuous learning from expert corrections

## Contributing

### Adding New Features

1. **Page Structure**: Follow existing page template structure
2. **State Management**: Use `st.session_state` for data persistence
3. **Error Handling**: Implement comprehensive error handling
4. **Documentation**: Update README with new features

### Model Updates

1. **Model Replacement**: Replace models in `fossil_classifier_final/`
2. **Version Control**: Maintain model version tracking
3. **Backward Compatibility**: Ensure interface compatibility
4. **Performance Testing**: Validate new model performance

## Technical Specifications

### System Requirements

- **Python**: 3.8+
- **TensorFlow**: 2.13+
- **Streamlit**: 1.25+
- **Memory**: 8GB+ RAM recommended
- **GPU**: Optional but recommended for faster inference

### Browser Support

- **Chrome**: 90+ (recommended)
- **Firefox**: 85+
- **Safari**: 14+
- **Edge**: 90+

### Performance Benchmarks

- **Model Loading**: ~5-10 seconds initial load
- **Image Processing**: ~50-200ms per image
- **UI Responsiveness**: <100ms for most interactions
- **Concurrent Users**: Up to 10 users (single instance)

This dashboard represents the culmination of the entire fossil classification pipeline, providing an accessible interface for utilizing the sophisticated AI models developed in the previous sections.
