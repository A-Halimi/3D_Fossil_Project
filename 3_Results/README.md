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

# View best model performance
cat convnext/reports/classification_report.txt

# Check training curves
python -c "
import json
import matplotlib.pyplot as plt

with open('convnext/history/training_history.json', 'r') as f:
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

#### 2. Compare Model Performance

```bash
# Generate comprehensive comparison report
cd ../2_AI_Modeling_Transfer_Learning
python fossil_model_compare.py --results_root ../3_Results --output_dir ../3_Results/_comparison

# View comparison results
cd ../3_Results/_comparison
cat model_comparison_report.md
```

#### 3. Load and Test Models

```python
# Example: Load and test the final ensemble model
import os, json, pathlib, numpy as np, tensorflow as tf, cv2
from tensorflow.keras import mixed_precision
from skimage import filters, morphology, measure
import matplotlib.pyplot as plt

# Setup environment
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mixed_precision.set_global_policy("mixed_float16")

# Get project root and model paths
PROJECT_ROOT = pathlib.Path.cwd().parent  # Adjust as needed
DEPLOY_DIR = PROJECT_ROOT / "3_Results" / "fossil_classifier_final"
MODEL_PATH = DEPLOY_DIR / "model.keras"
CLASSES_PATH = DEPLOY_DIR / "class_names.json"

# Load the final ensemble model and class names
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, 'r') as f:
    class_names = json.load(f)

# Advanced segmentation function for preprocessing
def segment_image(image_path, assume_bright_fossil=True, invert_output=False):
    """Segment fossil from background using Otsu thresholding"""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Apply Gaussian blur and Otsu thresholding
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        threshold = filters.threshold_otsu(blurred)
        binary = blurred > threshold if assume_bright_fossil else blurred < threshold
        
        # Clean up segmentation
        cleaned = morphology.remove_small_objects(binary, min_size=200)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=100)
        cleaned = morphology.binary_closing(cleaned, morphology.disk(2))
        
        # Extract main fossil region
        labeled = measure.label(cleaned)
        if labeled.max() == 0:
            return None
        
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

# Prediction function
def predict_fossil(image_path, use_clahe=False):
    """Predict fossil species from image path"""
    # Load and preprocess image
    original_img_bgr = cv2.imread(str(image_path))
    if original_img_bgr is None:
        print(f"Could not read image at {image_path}")
        return None
    
    # Check image brightness and apply appropriate segmentation
    img_gray_for_check = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2GRAY)
    if np.mean(img_gray_for_check) > 190:
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
    
    # Convert to RGB and create tensor
    final_rgb_np = cv2.cvtColor(segmented_np, cv2.COLOR_GRAY2RGB)
    img_tensor = tf.convert_to_tensor(final_rgb_np, dtype=tf.float32)
    
    # Apply the same cropping as used in training (crop_to_bbox)
    # Note: This assumes the PatchEnsemble model handles preprocessing internally
    
    # Make prediction
    predictions = model.predict(tf.expand_dims(img_tensor, 0))
    probs = predictions[0]
    
    # Get top 5 predictions
    top_5_indices = np.argsort(probs)[-5:][::-1]
    
    results = {
        'predicted_class': class_names[top_5_indices[0]],
        'confidence': float(probs[top_5_indices[0]]),
        'top_5_predictions': [
            {'class': class_names[idx], 'confidence': float(probs[idx])}
            for idx in top_5_indices
        ]
    }
    
    return results

# Example usage
image_path = "path/to/your/fossil/image.png"
result = predict_fossil(image_path, use_clahe=False)
if result:
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Top 5 predictions:")
    for pred in result['top_5_predictions']:
        print(f"  {pred['class']}: {pred['confidence']:.4f}")
```

## Key Metrics and Performance

### Model Performance Summary

Based on the comprehensive evaluation from `fossil_model_comparison_report_v2.md`:

#### Top Performing Models (Accuracy Leaderboard)
1. **ConvNeXt Large**: 95.12% accuracy (Top-3: 99.63%, AUC: 0.9978)
2. **NASNet**: 93.69% accuracy (Top-3: 99.15%, AUC: 0.9958)  
3. **EfficientNetV2 Large**: 93.53% accuracy (Top-3: 99.42%, AUC: 0.9957)
4. **EfficientNetV2 Small**: 91.82% accuracy (Top-3: 99.54%, AUC: 0.9956)
5. **ConvNeXt Base**: 91.34% accuracy (Top-3: 98.39%, AUC: 0.9926)
6. **MobileNet**: 88.02% accuracy (Top-3: 99.35%, AUC: 0.9949)
7. **ResNet101V2**: 84.34% accuracy (Top-3: 97.40%, AUC: 0.9835)

#### Macro F1-Score Performance
1. **ConvNeXt Large**: 0.9406 (best overall balance)
2. **NASNet**: 0.9254 
3. **EfficientNetV2 Large**: 0.9198
4. **EfficientNetV2 Small**: 0.9061
5. **ConvNeXt Base**: 0.8770
6. **MobileNet**: 0.8602
7. **ResNet101V2**: 0.8083

#### Per-Species Performance Insights

**Consistently High Performers (>95% F1-Score across most models):**
- **Fallotia**: Best overall (99.69-99.89% F1 across models)
- **Ataxophragmium**: Excellent stability (98.49-99.36% F1)
- **Arumella**: Strong performance (92.09-98.86% F1)
- **Alveolina**: Reliable classification (96.18-98.39% F1)

**Moderate Performers (85-95% F1-Score):**
- **Lockhartia**: Good consistency (98.15-98.61% F1)
- **Coskinolina**: Stable across models (94.33-98.23% F1)
- **Chrysalidina**: Variable performance (67.92-97.07% F1)
- **Minoxia**: Model-dependent (59.90-98.86% F1)

**Challenging Species (Recognition Difficulties):**
- **Baculogypsina**: Consistently challenging (14.84-66.35% F1)
- **Orbitoides**: Moderate difficulty (59.52-85.30% F1)
- **Elphidiella**: Variable across models (76.70-98.53% F1)

### Evaluation Metrics

Each model is evaluated using:
- **Overall Accuracy**: Percentage of correctly classified samples
- **Per-Class Precision**: True positives / (True positives + False positives)
- **Per-Class Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown
- **Top-3 Accuracy**: Percentage where correct class is in top 3 predictions

## Model Comparison Features

### Automated Analysis Tools

The [`_comparison/`](_comparison/) directory contains the actual generated files:

#### Performance Reports
- **`fossil_model_comparison_report_v2.md`** - Comprehensive comparison report with detailed analysis
- **`fossil_model_comparison_report_v2.csv`** - Quantitative metrics comparison table
- **`fossil_model_per_class_metrics_v2.csv`** - Per-species performance breakdown for all models
- **`fossil_model_per_class_best_v2.csv`** - Best performing models identified per class
- **`fossil_model_per_class_worst_v2.csv`** - Challenging cases and model weaknesses per class
- **`fossil_model_strengths_v2.json`** - Structured analysis of model strengths and recommendations

#### Analysis Features
- **Accuracy Leaderboards**: Ranked comparison across accuracy, macro F1, weighted F1, and precision/recall
- **Per-Class Analysis**: Species-specific performance comparison across all 7 models
- **Statistical Significance**: Model performance comparisons with confidence intervals
- **Ensemble Recommendations**: Data-driven suggestions for model combination strategies

### Generated Insights from Comparison Analysis

#### Model Performance Tiers

**Tier 1 - Production Ready (>95% Accuracy):**
- **ConvNeXt Large**: 95.12% accuracy, best overall performance
- Exceptional macro F1 (0.9406) and balanced precision/recall

**Tier 2 - High Performance (93-95% Accuracy):**
- **NASNet**: 93.69% accuracy, excellent AUC (0.9958)
- **EfficientNetV2 Large**: 93.53% accuracy, efficiency leader

**Tier 3 - Deployment Options (90-93% Accuracy):**
- **EfficientNetV2 Small**: 91.82% accuracy, resource-efficient
- **ConvNeXt Base**: 91.34% accuracy, solid baseline

**Tier 4 - Specialized Use Cases (<90% Accuracy):**
- **MobileNet**: 88.02% accuracy, mobile/edge deployment
- **ResNet101V2**: 84.34% accuracy, legacy comparison baseline

#### Species-Specific Model Recommendations

**For Consistent High Performance:**
- **Fallotia**: All models excellent (>99.6% F1) - deployment flexible
- **Ataxophragmium**: ConvNeXt Large leads (99.23% F1) - use for critical applications

**For Challenging Species:**
- **Baculogypsina**: ConvNeXt Large best (65.25% F1) - requires ensemble approach
- **Orbitoides**: Multiple models competitive - ensemble recommended

**For Resource-Constrained Deployment:**
- **EfficientNetV2 Small**: Best accuracy/efficiency trade-off
- **MobileNet**: Minimum viable performance for mobile applications

### Key Insights from Comparison

#### Model Strengths by Architecture

**ConvNeXt Large - The Overall Champion:**
- **Best overall accuracy**: 95.12% with exceptional macro F1 (0.9406)
- **Consistent performance**: Leads in macro precision (0.9615) and recall (0.9366)
- **Reliable across species**: Strong performance on challenging classes
- **Production ready**: Best choice for high-accuracy requirements

**NASNet - The Balanced Performer:**
- **Second-best accuracy**: 93.69% with solid top-3 performance (99.15%)
- **Excellent AUC**: 0.9958 indicating strong probabilistic calibration
- **Research value**: Good comparative baseline for advanced architectures

**EfficientNetV2 Models - The Efficiency Leaders:**
- **Large variant**: 93.53% accuracy with excellent efficiency trade-off
- **Small variant**: 91.82% accuracy, ideal for resource-constrained deployment
- **High top-3 accuracy**: Both variants achieve >99.4% top-3 performance
- **Deployment friendly**: Best balance of accuracy and computational efficiency

**MobileNet - The Edge Champion:**
- **Mobile optimized**: 88.02% accuracy with minimal computational requirements
- **Surprising top-3**: 99.35% top-3 accuracy despite lower overall performance
- **Edge deployment**: Ideal for mobile applications and embedded systems

#### Species-Specific Model Strengths

**For Challenging Species (Baculogypsina, Orbitoides):**
- **ConvNeXt Large excels**: Best performance on difficult-to-classify species
- **EfficientNetV2 Small**: Surprisingly competitive on Baculogypsina (66.35% F1)
- **Ensemble opportunity**: Different models struggle with different aspects

**For High-Performance Species (Fallotia, Ataxophragmium):**
- **Universal success**: All models perform well (>95% F1)
- **ConvNeXt Large leads**: Marginal but consistent advantage
- **Deployment flexibility**: Multiple viable options for these species

#### Practical Deployment Insights

**Production Deployment Recommendations:**
- **High accuracy**: Use ConvNeXt Large (95.12% accuracy)
- **Balanced deployment**: EfficientNetV2 Large (93.53% accuracy, better efficiency)
- **Resource-constrained**: EfficientNetV2 Small (91.82% accuracy, fast inference)
- **Mobile/edge**: MobileNet (88.02% accuracy, minimal resources)

**Ensemble Opportunities:**
- **ConvNeXt Large + EfficientNetV2**: Complementary strengths for challenging species
- **Class-specific routing**: Use best model per species for optimal performance
- **Confidence-based switching**: Leverage different models based on prediction confidence


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



