# 2_AI_Modeling_Transfer_Learning

## Overview

This section implements deep learning models for 3D fossil classification using transfer learning. It includes multiple state-of-the-art CNN architectures, ensemble methods, and comprehensive model comparison tools. The models are trained on the segmented dataset from Section 1 to classify 12 different fossil species.

## Purpose

- Train multiple CNN architectures using transfer learning
- Implement advanced data augmentation (CutMix/MixUp)
- Apply intelligent bounding-box cropping to focus on fossil regions
- Create ensemble models for improved accuracy
- Compare model performance across architectures
- Generate comprehensive evaluation reports
- Deploy production-ready models

## Contents

### Training Notebooks

- **[`DeepLearning_classification-convnext_segmented.ipynb`](DeepLearning_classification-convnext_segmented.ipynb)** - ConvNeXt model training
- **[`DeepLearning_classification-convnextl_segmented.ipynb`](DeepLearning_classification-convnextl_segmented.ipynb)** - ConvNeXt Large model
- **[`DeepLearning_classification-effv2l_segmented.ipynb`](DeepLearning_classification-effv2l_segmented.ipynb)** - EfficientNetV2 Large
- **[`DeepLearning_classification-effv2s_segmented.ipynb`](DeepLearning_classification-effv2s_segmented.ipynb)** - EfficientNetV2 Small  
- **[`DeepLearning_classification-mobilnet_segmented.ipynb`](DeepLearning_classification-mobilnet_segmented.ipynb)** - MobileNet training
- **[`DeepLearning_classification-nasnet_segmented.ipynb`](DeepLearning_classification-nasnet_segmented.ipynb)** - NASNet training
- **[`DeepLearning_classification-resnet101v2_segmented.ipynb`](DeepLearning_classification-resnet101v2_segmented.ipynb)** - ResNet101V2 training

### Ensemble & Analysis

- **[`DeepLearning_classification-ensemble_weighted_segmented-final.ipynb`](DeepLearning_classification-ensemble_weighted_segmented-final.ipynb)** - Final ensemble model creation and deployment

### Model Comparison Tools

- **[`fossil_model_compare.py`](fossil_model_compare.py)** - Comprehensive model comparison and reporting script

### Sample Images

- **[`Arumella_specimen2.png`](Arumella_specimen2.png)** - Sample fossil image
- **[`avizo_*.png`](.)** - Additional sample images for testing

## Supported Architectures

### Individual Models
1. **ConvNeXt** (Base & Large) - Modern CNN with transformer-inspired design
2. **EfficientNetV2** (Small & Large) - Efficient scaling of CNNs
3. **MobileNet** - Lightweight architecture for mobile deployment
4. **NASNet** - Neural Architecture Search optimized network
5. **ResNet101V2** - Deep residual network with improved design

### Ensemble Methods
- **Weighted Ensemble** - Combines ConvNeXt Large + EfficientNetV2 Small
- **Adaptive Weighting** - Dynamic weighting based on class difficulty

## Getting Started

### Prerequisites

Ensure you're in the recommended Docker environment and have completed Section 1:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -p 10000:8888 -p 8501:8501 -v ${PWD}:/workspace/mycode abdelghafour1/ngc_tf_rapids_25_01_vscode_torch:2025-v3 jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://$(hostname):10000
```

### Step-by-Step Training Process

#### 1. Verify Dataset

```bash
# Ensure dataset from Section 1 exists
ls ../1_Dataset_Creation/3d_fossil_dataset_segmented_final/
```

#### 2. Train Individual Models

```bash
cd 2_AI_Modeling_Transfer_Learning

# Start with a lightweight model for testing
jupyter lab DeepLearning_classification-mobilnet_segmented.ipynb

# Then train more complex models
jupyter lab DeepLearning_classification-convnext_segmented.ipynb
jupyter lab DeepLearning_classification-effv2s_segmented.ipynb
```

#### 3. Advanced Models (GPU Intensive)

```bash
# Large models - require significant GPU memory
jupyter lab DeepLearning_classification-convnextl_segmented.ipynb
jupyter lab DeepLearning_classification-effv2l_segmented.ipynb
jupyter lab DeepLearning_classification-resnet101v2_segmented.ipynb
jupyter lab DeepLearning_classification-nasnet_segmented.ipynb
```

#### 4. Create Ensemble Model

```bash
# After training individual models
jupyter lab DeepLearning_classification-ensemble_weighted_segmented-final.ipynb
```

#### 5. Compare Model Performance

```bash
# Generate comprehensive comparison report
python fossil_model_compare.py --results_root ../3_Results --recall_gap 0.15 --precision_gap 0.15
```

## Training Configuration

### Common Parameters (Configurable in each notebook)

```python
# Data configuration
DATA_ROOT = PROJECT_ROOT / "1_Dataset_Creation" / "3d_fossil_dataset_segmented_final"
BATCH_SIZE = 32                    # Adjust based on GPU memory
IMG_SIZE = (224, 224)              # Standard input size
IMG_SIZE_BIG = (384, 384)          # For larger models

# Training parameters
FREEZE_EPOCHS = 10                 # Transfer learning phase
FINETUNE_EPOCHS = 20               # Fine-tuning phase
LEARNING_RATE = 1e-4               # Initial learning rate

# Data augmentation
CUTMIX_PROB = 0.5                  # CutMix probability
MIXUP_ALPHA = 0.2                  # MixUp parameter
```

### Advanced Features

#### Intelligent Preprocessing Pipeline

**Bounding-Box Crop System:**
The DeepLearning_classification notebooks implement an intelligent bounding-box cropping system that significantly improves model performance:

```python
def get_bounding_box_crop(image):
    """
    Automatically detects and crops to fossil content regions
    - Converts to grayscale for content detection
    - Identifies non-black pixels (fossil content)
    - Calculates tight bounding box around content
    - Crops image to remove unnecessary background
    - Resizes to target dimensions while preserving aspect ratio
    """
```

**Benefits of Bounding-Box Cropping:**
- **Enhanced Focus**: Eliminates distracting black background regions
- **Improved Signal-to-Noise**: Maximizes fossil content per pixel
- **Better Feature Learning**: Forces models to focus on morphological details
- **Consistent Framing**: Normalizes fossil positioning across images
- **Reduced Computational Load**: Fewer irrelevant pixels to process

#### Data Augmentation Pipeline

```python
# Each notebook includes sophisticated augmentation:
- Intelligent bounding-box cropping to focus on fossil content
- CutMix/MixUp for improved generalization
- Rotation, scaling, and flipping
- Color jittering and contrast adjustment
- Smart cropping to maximize fossil region coverage
```


#### Transfer Learning Strategy

```python
# Two-phase training approach:
1. Freeze backbone + train classifier (FREEZE_EPOCHS)
2. Unfreeze layers + fine-tune (FINETUNE_EPOCHS)
```

## Model Performance Tracking

### Automatic Logging

Each notebook automatically:
- Saves best model checkpoints
- Generates classification reports
- Creates confusion matrices
- Logs training history
- Exports model predictions

### Output Structure

```
../3_Results/
├── convnext/
│   ├── ckpts/best.keras
│   ├── reports/classification_report.txt
│   └── history/training_log.json
├── effv2s/
│   └── ...
└── ensemble/
    ├── model.keras
    └── class_names.json
```

## Model Comparison Features

The [`fossil_model_compare.py`](fossil_model_compare.py) script provides:

### Comprehensive Analysis
- **Overall Performance**: Accuracy, F1-scores, precision, recall
- **Per-Class Analysis**: Best/worst performing models per species
- **Statistical Comparison**: Confidence intervals and significance tests
- **Strength/Weakness Mapping**: Identifies model specializations

### Usage Example

```bash
python fossil_model_compare.py \
    --results_root ../3_Results \
    --recall_gap 0.15 \
    --precision_gap 0.15 \
    --output_dir ../3_Results/_comparison
```

### Generated Reports
- **Markdown Report**: Human-readable comparison summary
- **CSV Files**: Machine-readable metrics for further analysis
- **JSON Summary**: Programmatic access to results

## Ensemble Model Details

The final ensemble in [`DeepLearning_classification-ensemble_weighted_segmented-final.ipynb`](DeepLearning_classification-ensemble_weighted_segmented-final.ipynb):

### Architecture
- **Main Model**: ConvNeXt Large (384×384 input)
- **Patch Model**: EfficientNetV2 Small (224×224 input)
- **Weighting Strategy**: Adaptive based on class difficulty

### Special Features
- **Weak Class Boosting**: Enhanced performance for difficult species
- **Multi-Scale Processing**: Different input sizes for different models
- **Production Deployment**: Optimized for inference speed

## Connection to Other Sections

- **Input from Section 1**: Uses `3d_fossil_dataset_segmented_final/` for training
- **Output to Section 3**: Saves trained models and evaluation results
- **Integration with Section 4**: Provides models for dashboard deployment

## GPU Requirements

### Minimum Requirements
- **VRAM**: 8GB for small models (MobileNet, EfficientNetV2-S)
- **VRAM**: 32GB+ for large models (ConvNeXt-L, EfficientNetV2-L)
- **RAM**: 32GB+ system memory recommended

### Training Time Estimates
- **Small Models**: 2-4 hours per model
- **Large Models**: 8-12 hours per model
- **Ensemble Training**: 1-2 hours (after individual models)

## Tips for New Users

1. **Start Small**: Begin with MobileNet to verify setup
2. **Monitor GPU Memory**: Use `nvidia-smi` to track VRAM usage
3. **Checkpoint Strategy**: Models auto-save best weights during training
4. **Hyperparameter Tuning**: Adjust batch size based on available memory
5. **Data Validation**: Verify dataset loading before starting long training runs

## Advanced Configuration

### Custom Model Registry

```python
# Each notebook includes a model registry for easy switching:
REGISTRY = {
    "convnext": (ConvNeXtBase, convnext.preprocess_input),
    "effv2s": (EfficientNetV2S, efficientnet_v2.preprocess_input),
    # ... add custom models here
}
```

### Training Optimization

```python
# Memory optimization settings:
mixed_precision.set_global_policy("mixed_float16")  # Faster training
tf.config.experimental.set_memory_growth(gpu, True)  # Dynamic memory
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Verify GPU utilization with `nvidia-smi`
3. **Poor Convergence**: Check learning rate and data augmentation settings
4. **Model Loading Errors**: Ensure TensorFlow version compatibility

### Performance Optimization

```python
# Optimize data pipeline:
AUTOTUNE = tf.data.AUTOTUNE
dataset = dataset.prefetch(AUTOTUNE).cache()
```

## Next Steps

After training models, proceed to:
1. **[3_Results](../3_Results/README.md)** - Analyze and compare model performance
2. **[4_Dashboard_App](../4_Dashboard_App/README.md)** - Deploy models in interactive dashboard

## Production Deployment

The final ensemble model is automatically prepared for deployment:
- **Model File**: `model.keras` (TensorFlow SavedModel format)
- **Class Names**: `class_names.json` (species label mapping)
- **Preprocessing**: Standardized normalization pipeline
