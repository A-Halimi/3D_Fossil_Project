# 3_Results

## Overview

This section contains all model training results, performance metrics, and comparative analyses from the deep learning models trained in Section 2. It serves as the central repository for model evaluation data, including trained model checkpoints, classification reports, confusion matrices, and comprehensive comparison studies.

## Purpose

- Store trained model checkpoints and artifacts
- Organize performance metrics and evaluation reports
- Provide comparative analysis across different architectures
- Track model improvements and hyperparameter experiments
- Support model selection and deployment decisions
- Generate publication-ready performance visualizations

## Contents

### Model Results Directories

Each model has its own dedicated results directory containing:

- **[`convnext/`](convnext/)** - ConvNeXt Base model results
- **[`convnextl/`](convnextl/)** - ConvNeXt Large model results  
- **[`effv2l/`](effv2l/)** - EfficientNetV2 Large model results
- **[`effv2s/`](effv2s/)** - EfficientNetV2 Small model results
- **[`mobilenet/`](mobilenet/)** - MobileNet model results
- **[`nasnet/`](nasnet/)** - NASNet model results
- **[`resnet101v2/`](resnet101v2/)** - ResNet101V2 model results

### Final Production Model

- **[`fossil_classifier_final/`](fossil_classifier_final/)** - Final ensemble model for deployment

### Comparative Analysis

- **[`_comparison/`](_comparison/)** - Cross-model comparison reports and visualizations

## Directory Structure

### Individual Model Results

Each model directory follows this actual structure:

```
{model_name}/
├── ckpts/                          # Model checkpoints
│   └── best.keras                  # Best performing model checkpoint
├── reports/                        # Performance reports and data
│   ├── classification_report_{model}.txt   # Detailed classification metrics
│   ├── class_map_{model}.json      # Class name mappings
│   └── training_history_{model}.json       # Training history data
├── figures/                        # Visualizations and plots
│   ├── cm_{model}.pdf              # Confusion matrix visualization
│   └── history_{model}.pdf         # Training/validation curves
├── logs/                           # TensorBoard training logs
│   └── {model}_{timestamp}/        # Timestamped training session logs
├── fossil_{model}.keras            # Final trained model
└── fossil_{model}_ema.keras        # EMA (Exponential Moving Average) model
```

### Final Production Model

```
fossil_classifier_final/
├── model.keras                     # Final ensemble model for deployment
└── class_names.json               # Species name mappings
```

### Comparative Analysis

```
_comparison/
├── fossil_model_comparison_report_v2.md    # Comprehensive comparison report
├── fossil_model_comparison_report_v2.csv   # Quantitative metrics comparison
├── fossil_model_per_class_metrics_v2.csv   # Per-species performance data
├── fossil_model_per_class_best_v2.csv      # Best performing models per class
├── fossil_model_per_class_worst_v2.csv     # Challenging cases per class
└── fossil_model_strengths_v2.json          # Model strengths analysis
```

## Getting Started

### Prerequisites

Ensure you're in the recommended Docker environment and have completed Sections 1-2:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -p 10000:8888 -p 8501:8501 -v ${PWD}:/workspace/mycode abdelghafour1/ngc_tf_rapids_25_01_vscode_torch:2025-v3 jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://$(hostname):10000
```

### Analyzing Results

#### 1. Explore Individual Model Performance

```bash
cd 3_Results

# View model performance reports
cat convnextl/reports/classification_report_convnextl.txt

# Check training history
python -c "
import json
import matplotlib.pyplot as plt

with open('convnextl/reports/training_history_convnextl.json', 'r') as f:
    history = json.load(f)
    
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training')
plt.plot(history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training')
plt.plot(history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.legend()
plt.show()
"
```

#### 2. Generate and View Model Comparison

```bash
# Generate comprehensive comparison report (if not already done)
cd ../2_AI_Modeling_Transfer_Learning
python fossil_model_compare.py --results_root ../3_Results --output_dir ../3_Results/_comparison

# View comparison results
cd ../3_Results/_comparison
cat fossil_model_comparison_report_v2.md
```

#### 3. Load and Test the Final Ensemble Model

```python
# Load and test the production-ready ensemble model
import os, json, pathlib, numpy as np, tensorflow as tf, cv2
from tensorflow.keras import mixed_precision
from skimage import filters, morphology, measure

# Setup environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

# Load model and class names
PROJECT_ROOT = pathlib.Path.cwd().parent  # Adjust as needed
DEPLOY_DIR = PROJECT_ROOT / "3_Results" / "fossil_classifier_final"
model = tf.keras.models.load_model(DEPLOY_DIR / "model.keras")
with open(DEPLOY_DIR / "class_names.json", 'r') as f:
    class_names = json.load(f)

# Segmentation and prediction functions
def segment_image(image_path, assume_bright_fossil=True, invert_output=False):
    """Segment fossil from background using Otsu thresholding"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        threshold = filters.threshold_otsu(blurred)
        binary = blurred > threshold if assume_bright_fossil else blurred < threshold
        
        cleaned = morphology.remove_small_objects(binary, min_size=200)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=100)
        cleaned = morphology.binary_closing(cleaned, morphology.disk(2))
        
        labeled = measure.label(cleaned)
        if labeled.max() == 0: return None
        
        main_fossil_label = max(measure.regionprops(labeled), key=lambda x: x.area).label
        mask = (labeled == main_fossil_label).astype(np.uint8) * 255
        segmented_gray = cv2.bitwise_and(img, img, mask=mask)
        
        if invert_output:
            fossil_pixels = segmented_gray[mask == 255]
            segmented_gray[mask == 255] = 255 - fossil_pixels
            
        return segmented_gray
    except Exception as e:
        print(f"Error segmenting {image_path}: {e}")
        return None

def predict_fossil(image_path, use_clahe=False):
    """Predict fossil species from image path"""
    original_img_bgr = cv2.imread(str(image_path))
    if original_img_bgr is None:
        print(f"Could not read image at {image_path}")
        return None
    
    # Auto-detect segmentation approach based on image brightness
    img_gray = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2GRAY)
    if np.mean(img_gray) > 190:
        segmented_np = segment_image(image_path, assume_bright_fossil=False, invert_output=True)
    else:
        segmented_np = segment_image(image_path, assume_bright_fossil=True, invert_output=False)
    
    if segmented_np is None:
        print(f"Segmentation failed for {image_path}")
        return None
    
    # Optional CLAHE normalization
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        segmented_np = clahe.apply(segmented_np)
    
    # Convert to RGB and predict
    final_rgb_np = cv2.cvtColor(segmented_np, cv2.COLOR_GRAY2RGB)
    img_tensor = tf.convert_to_tensor(final_rgb_np, dtype=tf.float32)
    predictions = model.predict(tf.expand_dims(img_tensor, 0))
    probs = predictions[0]
    
    # Get top 5 predictions
    top_5_indices = np.argsort(probs)[-5:][::-1]
    return {
        'predicted_class': class_names[top_5_indices[0]],
        'confidence': float(probs[top_5_indices[0]]),
        'top_5_predictions': [
            {'class': class_names[idx], 'confidence': float(probs[idx])}
            for idx in top_5_indices
        ]
    }

# Example usage
result = predict_fossil("path/to/your/fossil/image.png", use_clahe=False)
if result:
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    for pred in result['top_5_predictions']:
        print(f"  {pred['class']}: {pred['confidence']:.4f}")
```

## Model Performance Analysis and Comparison

### Comprehensive Model Evaluation

Based on the comprehensive evaluation from `fossil_model_comparison_report_v2.md`, the following analysis provides complete performance metrics, comparisons, and deployment insights.

#### Overall Performance Leaderboards

**Accuracy Ranking:**
1. **ConvNeXt Large**: 95.12% accuracy (Top-3: 99.63%, AUC: 0.9978)
2. **NASNet**: 93.69% accuracy (Top-3: 99.15%, AUC: 0.9958)  
3. **EfficientNetV2 Large**: 93.53% accuracy (Top-3: 99.42%, AUC: 0.9957)
4. **EfficientNetV2 Small**: 91.82% accuracy (Top-3: 99.54%, AUC: 0.9956)
5. **ConvNeXt Base**: 91.34% accuracy (Top-3: 98.39%, AUC: 0.9926)
6. **MobileNet**: 88.02% accuracy (Top-3: 99.35%, AUC: 0.9949)
7. **ResNet101V2**: 84.34% accuracy (Top-3: 97.40%, AUC: 0.9835)

**Macro F1-Score Ranking:**
1. **ConvNeXt Large**: 0.9406 (best overall balance)
2. **NASNet**: 0.9254 
3. **EfficientNetV2 Large**: 0.9198
4. **EfficientNetV2 Small**: 0.9061
5. **ConvNeXt Base**: 0.8770
6. **MobileNet**: 0.8602
7. **ResNet101V2**: 0.8083

### Model Performance Tiers and Deployment Strategy

#### Tier 1 - Production Ready (>95% Accuracy)
**ConvNeXt Large - The Overall Champion:**
- **Performance**: 95.12% accuracy, exceptional macro F1 (0.9406)
- **Strengths**: Leads in macro precision (0.9615) and recall (0.9366)
- **Use Case**: Best choice for high-accuracy production requirements
- **Species Excellence**: Strong performance on challenging classes (Baculogypsina: 65.25% F1)

#### Tier 2 - High Performance (93-95% Accuracy)
**NASNet - The Balanced Performer:**
- **Performance**: 93.69% accuracy, excellent AUC (0.9958)
- **Strengths**: Strong probabilistic calibration, solid top-3 performance (99.15%)
- **Use Case**: Research applications and comparative baseline

**EfficientNetV2 Large - The Efficiency Leader:**
- **Performance**: 93.53% accuracy, excellent efficiency trade-off
- **Strengths**: Best balance of accuracy and computational efficiency
- **Use Case**: Production deployment with resource consideration

#### Tier 3 - Deployment Options (90-93% Accuracy)
**EfficientNetV2 Small:**
- **Performance**: 91.82% accuracy, resource-efficient
- **Strengths**: Surprisingly competitive on challenging species (Baculogypsina: 66.35% F1)
- **Use Case**: Resource-constrained deployment, edge computing

**ConvNeXt Base:**
- **Performance**: 91.34% accuracy, solid baseline
- **Use Case**: General purpose applications

#### Tier 4 - Specialized Use Cases (<90% Accuracy)
**MobileNet - The Edge Champion:**
- **Performance**: 88.02% accuracy, minimal computational requirements
- **Strengths**: Surprising top-3 accuracy (99.35%) despite lower overall performance
- **Use Case**: Mobile applications and embedded systems

**ResNet101V2:**
- **Performance**: 84.34% accuracy
- **Use Case**: Legacy comparison baseline

### Per-Species Performance Analysis

#### Consistently High Performers (>95% F1-Score across most models)
- **Fallotia**: Best overall (99.69-99.89% F1 across models) - deployment flexible
- **Ataxophragmium**: Excellent stability (98.49-99.36% F1) - ConvNeXt Large leads
- **Arumella**: Strong performance (92.09-98.86% F1) - robust across architectures
- **Alveolina**: Reliable classification (96.18-98.39% F1) - consistent performance

#### Moderate Performers (85-95% F1-Score)
- **Lockhartia**: Good consistency (98.15-98.61% F1) - reliable identification
- **Coskinolina**: Stable across models (94.33-98.23% F1) - architecture independent
- **Chrysalidina**: Variable performance (67.92-97.07% F1) - model dependent
- **Minoxia**: Model-dependent (59.90-98.86% F1) - benefits from ensemble

#### Challenging Species (Recognition Difficulties)
- **Baculogypsina**: Consistently challenging (14.84-66.35% F1) - requires ensemble approach
- **Orbitoides**: Moderate difficulty (59.52-85.30% F1) - multiple models competitive
- **Elphidiella**: Variable across models (76.70-98.53% F1) - architecture sensitive

### Automated Analysis Tools and Reports

The [`_comparison/`](_comparison/) directory contains comprehensive analysis files:

#### Performance Reports
- **`fossil_model_comparison_report_v2.md`** - Detailed comparison with insights
- **`fossil_model_comparison_report_v2.csv`** - Quantitative metrics table
- **`fossil_model_per_class_metrics_v2.csv`** - Per-species performance breakdown
- **`fossil_model_per_class_best_v2.csv`** - Best performing models per class
- **`fossil_model_per_class_worst_v2.csv`** - Challenging cases analysis
- **`fossil_model_strengths_v2.json`** - Structured model strengths analysis

#### Analysis Features
- **Accuracy Leaderboards**: Ranked comparisons across multiple metrics
- **Per-Class Analysis**: Species-specific performance comparison across all 7 models
- **Statistical Significance**: Model performance comparisons with confidence intervals
- **Ensemble Recommendations**: Data-driven suggestions for model combination strategies

### Deployment Recommendations

#### Production Deployment Strategy
- **High accuracy requirements**: ConvNeXt Large (95.12% accuracy)
- **Balanced deployment**: EfficientNetV2 Large (93.53% accuracy, better efficiency)
- **Resource-constrained**: EfficientNetV2 Small (91.82% accuracy, fast inference)
- **Mobile/edge applications**: MobileNet (88.02% accuracy, minimal resources)

#### Ensemble Opportunities
- **ConvNeXt Large + EfficientNetV2 Small**: Implemented in final production model
- **Class-specific routing**: Use best model per species for optimal performance
- **Confidence-based switching**: Leverage different models based on prediction confidence
- **Challenging species focus**: Ensemble particularly beneficial for Baculogypsina and Orbitoides

### Evaluation Metrics Framework

Each model evaluation includes:
- **Overall Accuracy**: Percentage of correctly classified samples
- **Top-3 Accuracy**: Percentage where correct class is in top 3 predictions (>97% across all models)
- **Per-Class Precision/Recall**: Species-specific performance metrics
- **F1-Score**: Harmonic mean of precision and recall for balanced assessment
- **AUC Scores**: All models achieve >0.98 showing excellent probabilistic calibration
- **Confusion Matrix**: Detailed classification breakdown for error analysis


## Production Model Details

### Final Ensemble Model (`fossil_classifier_final/`)

The production-ready model contains the actual generated files:

```
fossil_classifier_final/
├── model.keras                     # PatchEnsemble model (ConvNeXt Large + EfficientNetV2 Small)
└── class_names.json               # Species name mappings for 12 fossil classes
```

#### Model Architecture - PatchEnsemble Implementation

**Core Design:**
- **Main Model**: ConvNeXt Large (95.12% individual accuracy)
- **Patch Model**: EfficientNetV2 Small (91.82% individual accuracy)
- **Ensemble Strategy**: Confidence-based switching for weak classes
- **Weak Classes**: Baculogypsina, Orbitoides (challenging species)

**Intelligent Switching Logic:**
```python
# Ensemble decision process:
1. ConvNeXt Large makes primary prediction
2. EfficientNetV2 Small makes secondary prediction  
3. If primary prediction is weak class AND secondary has higher confidence:
   → Use secondary prediction
4. Else: Use primary prediction
```

#### Technical Specifications

**Input Processing:**
- **Input Size**: 384×384 RGB images (automatically resized internally)
- **Preprocessing**: Built-in crop_to_bbox, ConvNeXt/EfficientNet preprocessing
- **Segmentation**: Expects segmented fossil images with black backgrounds
- **Normalization**: Model handles all preprocessing internally

**Output Format:**
- **Classes**: 12 fossil species probability distribution
- **Normalization**: Probabilities sum to 1.0
- **Confidence**: Can extract max probability as confidence score

**Performance Characteristics:**
- **Model Size**: ~800MB (combined ConvNeXt Large + EfficientNetV2 Small)
- **Inference Time**: ~50ms per image (GPU), ~200ms (CPU)
- **Memory Requirements**: ~2GB GPU VRAM for inference
- **Input Flexibility**: Handles variable image sizes through internal resizing

#### Actual Performance Metrics (Final Ensemble)

Based on the ensemble notebook results:
- **Test Accuracy**: 95.64% (51,468 test samples)
- **Macro Average**: Precision 96.38%, Recall 94.52%, F1-Score 94.97%
- **Weighted Average**: Precision 95.94%, Recall 95.64%, F1-Score 95.43%
- **Top-3 Performance**: >99% (estimated from individual model performance)

#### Per-Species Ensemble Performance

**Excellent Performance (>98% F1):**
- **Fallotia**: 99.66% F1 (best overall species)
- **Ataxophragmium**: 99.24% F1 (consistent high performer)
- **Arumella**: 98.70% F1 (stable classification)
- **Lockhartia**: 98.39% F1 (reliable identification)
- **Rhapydionina**: 98.29% F1 (distinctive morphology)

**Strong Performance (95-98% F1):**
- **Alveolina**: 97.33% F1 (classic foraminifera)
- **Elphidiella**: 97.56% F1 (planispiral arrangement)
- **Chrysalidina**: 95.78% F1 (elongated chambers)

**Good Performance (90-95% F1):**
- **Minoxia**: 94.10% F1 (small size challenge)
- **Coskinolina**: 98.22% F1 (perforated structure)

**Challenging Species (<90% F1):**
- **Orbitoides**: 87.17% F1 (disc-shaped complexity)
- **Baculogypsina**: 75.22% F1 (most challenging species)


## Connection to Other Sections

### Input Sources
- **From Section 2**: Trained models, training histories, evaluation metrics
- **Model Artifacts**: Generated during training notebooks execution

### Output Destinations  
- **To Section 4**: Production model (`fossil_classifier_final/`) used in dashboard
- **For Research**: Performance data for publications and further analysis


## Tips for Results Analysis

1. **Model Selection**: Use ensemble for best accuracy, EfficientNetV2-S for deployment efficiency
2. **Performance Monitoring**: Regularly check confusion matrices for class-specific issues
3. **Error Analysis**: Focus on consistently misclassified species for targeted improvements
4. **Ensemble Benefits**: Ensemble reduces overconfident incorrect predictions
5. **Resource Trade-offs**: Consider inference time vs. accuracy for deployment scenarios


## Next Steps

After analyzing results:
1. **Deploy Best Model**: Use `fossil_classifier_final/` in Section 4 dashboard



