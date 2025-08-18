# 1_Dataset_Creation

## Overview

This section handles the creation and preprocessing of the 3D fossil dataset from raw NIfTI files. It extracts 2D slices from 3D fossil volumes, applies intelligent filtering to remove noise, and organizes the data into training/validation/test splits for machine learning.

## Purpose

- Convert 3D fossil NIfTI files (`.nii` format) into 2D slice datasets
- Apply smart thresholding and mask-guided sampling to focus on fossil content
- Create segmented versions with black backgrounds for improved model training
- Generate balanced datasets across 12 fossil species
- Organize data into proper train/validation/test splits

## Contents

### Core Scripts

- **[`dataset_creation.py`](dataset_creation.py)** - Main dataset creation script with intelligent slice sampling
- **[`segment_fossils_black_bg.py`](segment_fossils_black_bg.py)** - Applies segmentation to create black background versions
- **[`run_full_segmentation_black_bg.py`](run_full_segmentation_black_bg.py)** - Batch processing script for all species

### Notebooks

- **[`Dataset_creation_segmented_final.ipynb`](Dataset_creation_segmented_final.ipynb)** - Interactive notebook for dataset creation and analysis

### Configuration Files

- **[`Species_Analysis.txt`](Species_Analysis.txt)** - Comprehensive species selection analysis and dataset composition

### Input/Output Directories

- **`models/`** - Input directory containing raw NIfTI files organized by species
- **`3d_fossil_dataset_clean/`** - Output directory for the clean dataset
- **`3d_fossil_dataset_segmented_final/`** - Output directory for segmented dataset with black backgrounds

## Fossil Species Selection and Dataset Composition

### Species Selection Rationale

The final dataset focuses on **12 carefully selected fossil species** from an initial collection of **27 species across 97 3D models**. The selection criteria prioritized species with sufficient data for robust machine learning training.

#### Initial Species Distribution Analysis

From the comprehensive analysis in [`Species_Analysis.txt`](Species_Analysis.txt):

**Original Collection Statistics:**
- **Total species**: 27 unique foraminifera species
- **Total 3D models**: 97 high-resolution micro-CT scans
- **Species with single models**: 10 species (insufficient for ML training)
- **Species with 2-3 models**: 5 species (minimal data for robust training)
- **Species with 4+ models**: 12 species ✅ **Selected for ML pipeline**

#### Selected Species and Model Counts

The 12 selected species provide the optimal balance between diversity and data sufficiency:

| Species | 3D Models | Justification |
|---------|-----------|---------------|
| **Chrysalidina** | 16 models | Largest dataset, excellent for training stability |
| **Ataxophragmium** | 7 models | Good morphological variation |
| **Baculogypsina** | 7 models | Sufficient for train/val/test splits |
| **Minoxia** | 6 models | Adequate data for robust training |
| **Elphidiella** | 6 models | Represents distinct morphological group |
| **Fallotia** | 6 models | Key paleontological importance |
| **Arumella** | 5 models | Historical significance in classification |
| **Lockhartia** | 5 models | Distinctive morphological features |
| **Orbitoides** | 5 models | Represents large disc-shaped forms |
| **Rhapydionina** | 5 models | Unique conical morphology |
| **Alveolina** | 4 models | Classic foraminifera example |
| **Coskinolina** | 4 models | Distinctive perforated structure |

### Final Dataset Statistics

After processing through the complete pipeline, the segmented dataset (`3d_fossil_dataset_segmented_final/`) contains:

#### Dataset Composition by Species

| Species | Test | Training | Validation | **Total** |
|---------|------|----------|------------|-----------|
| Alveolina | 5,129 | 3,535 | 1,724 | **10,388** |
| Arumella | 2,163 | 3,597 | 1,143 | **6,903** |
| Ataxophragmium | 4,898 | 3,943 | 993 | **9,834** |
| Baculogypsina | 2,980 | 2,243 | 997 | **6,220** |
| Chrysalidina | 5,111 | 3,782 | 1,079 | **9,972** |
| Coskinolina | 5,060 | 3,350 | 1,583 | **9,993** |
| Elphidiella | 3,385 | 4,048 | 1,012 | **8,445** |
| Fallotia | 5,009 | 4,102 | 987 | **10,098** |
| Lockhartia | 5,359 | 3,899 | 1,234 | **10,492** |
| Minoxia | 3,557 | 4,156 | 1,034 | **8,747** |
| Orbitoides | 5,015 | 3,705 | 1,074 | **9,794** |
| Rhapydionina | 3,802 | 3,743 | 1,186 | **8,731** |
| **TOTAL** | **51,468** | **44,103** | **14,046** | **109,617** |

#### Dataset Balance Analysis

**Training Set Balance:**
- **Mean images per species**: 3,675 ± 511 slices
- **Range**: 2,243 - 4,156 slices per species
- **Coefficient of Variation**: 13.91% (excellent balance)
- **Total training images**: 44,103

**Validation Set Balance:**
- **Mean images per species**: 1,171 ± 241 slices  
- **Range**: 987 - 1,724 slices per species
- **Coefficient of Variation**: 20.58% (good balance)
- **Total validation images**: 14,046

**Test Set Balance:**
- **Mean images per species**: 4,289 ± 1,060 slices
- **Range**: 2,163 - 5,359 slices per species  
- **Coefficient of Variation**: 24.72% (acceptable for testing)
- **Total test images**: 51,468

#### Model Allocation Examples by Species

**Understanding the Process**: Each species follows a unique allocation pattern based on available 3D models and their fossil content richness.

| Species | Total Models | Test Models | Train Models | Val Models | Test Images | Train Images | Val Images |
|---------|--------------|-------------|--------------|------------|-------------|--------------|------------|
| **Chrysalidina** | 16 | 3 | 10 | 3 | 5,111 | 3,782 | 1,079 |
| **Ataxophragmium** | 7 | 2 | 4 | 1 | 4,898 | 3,943 | 993 |
| **Baculogypsina** | 7 | 2 | 4 | 1 | 2,980 | 2,243 | 997 |
| **Elphidiella** | 6 | 1 | 4 | 1 | 3,385 | 4,048 | 1,012 |
| **Alveolina** | 4 | 1 | 2 | 1 | 5,129 | 3,535 | 1,724 |
| **Coskinolina** | 4 | 1 | 2 | 1 | 5,060 | 3,350 | 1,583 |

**Key Insights from Model Allocation:**

1. **Chrysalidina Example**: 16 models → rich dataset allows 3-way split with good representation
2. **Alveolina Example**: 4 models → minimal models but high fossil content per model
3. **Variable Outcomes**: Same model count can yield different slice counts due to fossil density

**Why This Approach is Superior:**
- **Prevents Overfitting**: Models can't memorize individual specimen characteristics
- **Real-World Performance**: Test accuracy reflects performance on genuinely new fossils
- **Research Validity**: Maintains scientific rigor required for paleontological studies

#### Data Quality Metrics

- **Total high-quality slices**: 109,617 images
- **Average slices per species**: 9,135 images
- **Image format**: 224×224 RGB PNG with black backgrounds
- **Preprocessing**: Otsu thresholding, intelligent sampling, segmentation

#### Advanced Model-Level Splitting Methodology

**Critical Data Integrity Feature**: The dataset splitting is performed at the **3D model level** before slice extraction to prevent **data leakage**. This ensures that slices from the same fossil specimen never appear in different splits.

**Two-Stage Splitting Process:**

**Stage 1: Model-Level Split (Dataset Creation)**
1. **Initial 3D Model Allocation**: Each species' 3D models are divided using a ~20% test split
2. **Stratified Slice Generation**: Target of 10,000 slices per species from allocated models
3. **Quality Filtering**: Intelligent noise removal and fossil content validation

**Stage 2: Train/Validation Split (Model-Level)**
1. **Training Set Subdivision**: Training models are further split at model level
2. **Validation Target**: ~1,000 slices per species from dedicated validation models
3. **Model-Level Preservation**: Complete models assigned to either train or validation (never mixed)

**Concrete Example - Chrysalidina Species:**
- **Original**: 16 total 3D models
- **Test Split**: 3 models → 10,000 target slices → **5,111 final images** (after quality filtering)
- **Training Pool**: 13 models → 10,000 target slices → **4,861 final images**
- **Train/Val Subdivision**: 
  - **Validation**: 3 models → **1,079 images**
  - **Training**: 10 models → **3,782 images**

**Why Slice Counts Don't Match Split Percentages:**
The final image counts depend on:
1. **Variable fossil content per model**: Some 3D scans contain more/fewer fossil-rich slices
2. **Quality filtering**: Removal of empty or low-content slices varies by specimen
3. **Model-level constraints**: Split ratios are determined by whole models, not target percentages

**Data Integrity Guarantees:**
- ✅ **No Data Leakage**: Slices from same fossil never cross split boundaries
- ✅ **Realistic Evaluation**: Test performance reflects true generalization to new specimens
- ✅ **Scientific Validity**: Maintains specimen-level independence crucial for paleontological applications

## Getting Started

### Prerequisites

Ensure you're running in the recommended Docker environment:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -p 10000:8888 -p 8501:8501 -v ${PWD}:/workspace/mycode abdelghafour1/ngc_tf_rapids_25_01_vscode_torch:2025-v3 jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://$(hostname):10000
```

### Step-by-Step Usage

#### 1. Prepare Input Data

```bash
# Ensure your NIfTI files are organized in the models/ directory:
# models/
#   ├── Alveolina_specimen1.nii
#   ├── Alveolina_specimen2.nii
#   ├── Arumella_specimen1.nii
#   └── ...
```

#### 2. Create Clean Dataset

```bash
cd 1_Dataset_Creation
python dataset_creation.py
```

**Configuration options in [`dataset_creation.py`](dataset_creation.py):**
- `TARGET_IMAGES_PER_SPECIES`: Number of slices per species (default: 10,000)
- `TEST_SPLIT_FRAC`: Test set fraction (default: 0.20)
- `SAVE_FORMAT`: Output format - "png" or "npy" (default: "png")

#### 3. Create Segmented Dataset

```bash
# For individual species
python segment_fossils_black_bg.py

# For all species (recommended)
python run_full_segmentation_black_bg.py
```

#### 4. Interactive Analysis

```bash
# Launch Jupyter and open the notebook
jupyter lab Dataset_creation_segmented_final.ipynb
```

## Key Features

### Intelligent Slice Sampling

- **Mask-guided sampling**: Only extracts slices containing significant fossil content
- **Area threshold**: Removes slices with <2% fossil pixels  
- **Smart thresholding**: Uses Otsu's method for automatic threshold detection

### Data Quality Assurance

- **Noise filtering**: Removes near-empty or low-content slices
- **Balanced sampling**: Ensures equal representation across species
- **Consistent preprocessing**: Standardized normalization and resizing
- **Model-level splitting**: Prevents data leakage by maintaining specimen-level independence
- **Quality-driven allocation**: Final slice counts reflect actual fossil content, not arbitrary percentages

### Output Structure

```
3d_fossil_dataset_segmented_final/
├── train_split/
│   ├── Alveolina/
│   ├── Arumella/
│   └── ...
├── val_split/
│   ├── Alveolina/
│   ├── Arumella/
│   └── ...
└── test/
    ├── Alveolina/
    ├── Arumella/
    └── ...
```

## Configuration Parameters

### In [`dataset_creation.py`](dataset_creation.py):

```python
NIFTI_DIR = "models"                    # Input directory
OUTPUT_DIR = "3d_fossil_dataset_clean"  # Output directory  
TARGET_IMAGES_PER_SPECIES = 10000       # Slices per species
TEST_SPLIT_FRAC = 0.20                  # Test set size
SAVE_FORMAT = "png"                     # Output format
```

## Connection to Other Sections

- **Input to Section 2**: The `3d_fossil_dataset_segmented_final/` directory serves as input for AI model training
- **Preprocessing Pipeline**: Establishes the data preprocessing standards used throughout the project
- **Species Mapping**: Defines the 12-class classification problem for the AI models

## Tips for New Users

1. **Species Selection**: The 12 selected species represent optimal balance between data sufficiency and morphological diversity
2. **Dataset Size**: Total of 109,617 high-quality images across all species provides robust training data
3. **Balance Quality**: Training set CV of 13.91% indicates excellent balance across species
4. **Storage Requirements**: Ensure sufficient disk space (~50GB for full dataset)
5. **Memory Usage**: Large NIfTI files require substantial RAM during processing
6. **Quality Check**: Review a few generated slices to verify proper segmentation
7. **Batch Processing**: Use [`run_full_segmentation_black_bg.py`](run_full_segmentation_black_bg.py) for efficient processing of all species
8. **Format Choice**: Use PNG for visualization, NPY for faster loading in ML pipelines

## Troubleshooting

- **Memory Issues**: Reduce `TARGET_IMAGES_PER_SPECIES` for smaller datasets
- **Missing Files**: Verify NIfTI files are properly named and in `models/` directory
- **Segmentation Problems**: Check threshold values in [`segment_fossils_black_bg.py`](segment_fossils_black_bg.py)
- **Disk Space**: Monitor available storage during dataset generation

## Next Steps

After completing dataset creation, proceed to **[2_AI_Modeling_Transfer_Learning](../2_AI_Modeling_Transfer_Learning/README.md)** for model training and evaluation.
