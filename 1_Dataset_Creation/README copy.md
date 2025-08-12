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

- **[`Species_Analysis.txt`](Species_Analysis.txt)** - Species information and analysis notes

### Input/Output Directories

- **`models/`** - Input directory containing raw NIfTI files organized by species
- **`3d_fossil_dataset_clean/`** - Output directory for the clean dataset
- **`3d_fossil_dataset_segmented_final/`** - Output directory for segmented dataset with black backgrounds

## Fossil Species Supported

The dataset includes 12 fossil species:
- Alveolina
- Arumella  
- Ataxophragmium
- Baculogypsina
- Chrysalidina
- Coskinolina
- Elphidiella
- Fallotia
- Lockhartia
- Minoxia
- Orbitoides
- Rhapydionina

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

1. **Storage Requirements**: Ensure sufficient disk space (~50GB for full dataset)
2. **Memory Usage**: Large NIfTI files require substantial RAM during processing
3. **Quality Check**: Review a few generated slices to verify proper segmentation
4. **Batch Processing**: Use [`run_full_segmentation_black_bg.py`](run_full_segmentation_black_bg.py) for efficient processing of all species
5. **Format Choice**: Use PNG for visualization, NPY for faster loading in ML pipelines

## Troubleshooting

- **Memory Issues**: Reduce `TARGET_IMAGES_PER_SPECIES` for smaller datasets
- **Missing Files**: Verify NIfTI files are properly named and in `models/` directory
- **Segmentation Problems**: Check threshold values in [`segment_fossils_black_bg.py`](segment_fossils_black_bg.py)
- **Disk Space**: Monitor available storage during dataset generation

## Next Steps

After completing dataset creation, proceed to **[2_AI_Modeling_Transfer_Learning](../2_AI_Modeling_Transfer_Learning/README.md)** for model training and evaluation.
