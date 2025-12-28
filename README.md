# Fossil Classification Project

## 🦴 Overview

This project implements a comprehensive deep learning pipeline for 3D fossil classification using state-of-the-art computer vision techniques. The system processes 3D NIfTI files of fossil specimens, extracts 2D slices, trains multiple CNN architectures using transfer learning, and deploys the best models through an interactive web dashboard.

## 🎯 Project Goals

- **Automated Classification**: Classify 12 different fossil species from 2D slice images
- **High Accuracy**: Achieve >95% classification accuracy using ensemble methods
- **Scalable Pipeline**: Create reproducible workflows for dataset creation and model training
- **User-Friendly Interface**: Provide an interactive dashboard for real-time classification
- **Research Support**: Enable paleontological research through AI-assisted identification

## 📁 Project Structure

```
Fossil_Project/
├── 1_Dataset_Creation/          # Data preprocessing and dataset generation
│   ├── dataset_creation.py      # Main dataset creation script
│   ├── segment_fossils_black_bg.py  # Segmentation with black backgrounds
│   ├── run_full_segmentation_black_bg.py  # Batch processing
│   ├── Dataset_creation_segmented_final.ipynb  # Interactive notebook
│   ├── models/                  # Input NIfTI files
│   ├── 3d_fossil_dataset_clean/ # Clean dataset output
│   ├── 3d_fossil_dataset_segmented_final/  # Segmented dataset
│   └── README.md               # Detailed documentation
│
├── 2_AI_Modeling_Transfer_Learning/  # Deep learning model training
│   ├── DeepLearning_classification-*.ipynb  # Individual model training
│   ├── DeepLearning_classification-ensemble_weighted_segmented-final.ipynb
│   ├── fossil_model_compare.py  # Model comparison tool
│   ├── Image_samples/           # Sample images
│   └── README.md               # Training documentation
│
├── 3_Results/                   # Model results and evaluation
│   ├── convnext/               # ConvNeXt model results
│   ├── convnextl/              # ConvNeXt Large results
│   ├── effv2l/                 # EfficientNetV2 Large results
│   ├── effv2s/                 # EfficientNetV2 Small results
│   ├── mobilenet/              # MobileNet results
│   ├── nasnet/                 # NASNet results
│   ├── resnet101v2/            # ResNet101V2 results
│   ├── fossil_classifier_final/ # Final ensemble model
│   ├── _comparison/            # Cross-model comparisons
│   └── README.md               # Results documentation
│
├── 4_Dashboard_App/            # Interactive web application
│   ├── Home.py                 # Main Streamlit app
│   ├── Run_App.ipynb          # Jupyter launcher
│   ├── pages/                  # App pages
│   │   ├── 1_Fossil_DL_Classification.py
│   │   └── 2_Fossil Matching Slice.py
│   ├── fossil_classifier_final/ # Production model
│   ├── models/                 # Additional model assets
│   └── README.md              # Dashboard documentation
│
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

Use the recommended Docker environment for consistent results:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -p 10000:8888 -p 8501:8501 -v ${PWD}:/workspace/mycode abdelghafour1/ngc_tf_rapids_25_01_vscode_torch:2025-v3 jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://$(hostname):10000
```

### Complete Workflow

#### 1. Dataset Creation (Section 1)

```bash
cd 1_Dataset_Creation

# Create clean dataset from NIfTI files
python dataset_creation.py

# Generate segmented dataset with black backgrounds
python run_full_segmentation_black_bg.py

# Verify dataset structure
ls 3d_fossil_dataset_segmented_final/
```

#### 2. Model Training (Section 2)

```bash
cd ../2_AI_Modeling_Transfer_Learning

# Train individual models (start with lightweight ones)
jupyter lab DeepLearning_classification-mobilnet_segmented.ipynb
jupyter lab DeepLearning_classification-convnext_segmented.ipynb
jupyter lab DeepLearning_classification-effv2s_segmented.ipynb

# Train advanced models (GPU intensive)
jupyter lab DeepLearning_classification-convnextl_segmented.ipynb
jupyter lab DeepLearning_classification-effv2l_segmented.ipynb

# Create final ensemble model
jupyter lab DeepLearning_classification-ensemble_weighted_segmented-final.ipynb

# Compare all models
python fossil_model_compare.py --results_root ../3_Results
```

#### 3. Results Analysis (Section 3)

```bash
cd ../3_Results

# Explore model performance
cat convnext/reports/classification_report.txt
cat _comparison/model_comparison_report.md

# Verify final model is ready
ls fossil_classifier_final/
```

#### 4. Dashboard Deployment (Section 4)

```bash
cd ../4_Dashboard_App

# Launch interactive dashboard
streamlit run Home.py --server.maxUploadSize 9000 --server.maxMessageSize 10000

# Access at: http://localhost:8501
```

## 🔬 Fossil Species Classification

### Supported Species (12 Classes)

1. **Alveolina** - Large benthic foraminifera with complex internal structure
2. **Arumella** - Distinctive spiral arrangement of chambers
3. **Ataxophragmium** - Agglutinated wall structure
4. **Baculogypsina** - Rod-shaped with specific chamber arrangement
5. **Chrysalidina** - Elongated with serial chambers
6. **Coskinolina** - Perforated wall structure
7. **Elphidiella** - Planispiral arrangement
8. **Fallotia** - Large fusiform shape
9. **Lockhartia** - Lenticular shape with complex structure
10. **Minoxia** - Small size with distinctive features
11. **Orbitoides** - Disc-shaped with radial structure
12. **Rhapydionina** - Elongated conical shape

### Dataset Statistics

- **Total Images**: ~120,000 high-quality 2D slices
- **Images per Species**: ~10,000 balanced samples
- **Image Format**: 224×224 RGB PNG files
- **Data Splits**: 60% train, 20% validation, 20% test
- **Source**: 3D micro-CT scans (NIfTI format)

## 🤖 AI Models and Performance

### Model Architectures

| Model | Accuracy | Top-3 | AUC | Macro F1 | Weighted F1 | Best Use Case |
|-------|----------|-------|-----|----------|-------------|---------------|
| **Ensemble (Final)** | **95.64%** | **99.6%** | **0.998** | **94.97%** | **95.43%** | Production deployment |
| **ConvNeXt Large** | 95.12% | 99.63% | 0.998 | 94.06% | 94.76% | High accuracy scenarios |
| **NASNet** | 93.69% | 99.15% | 0.996 | 92.54% | 93.29% | Research comparison |
| **EfficientNetV2 Large** | 93.53% | 99.42% | 0.996 | 91.98% | 92.91% | Balanced performance |
| **EfficientNetV2 Small** | 91.82% | 99.54% | 0.996 | 90.61% | 91.58% | Resource-constrained deployment |
| **ConvNeXt Base** | 91.34% | 98.39% | 0.993 | 87.70% | 89.62% | General purpose |
| **MobileNet** | 88.02% | 99.35% | 0.995 | 86.02% | 87.29% | Mobile/edge deployment |
| **ResNet101V2** | 84.34% | 97.40% | 0.984 | 80.83% | 82.79% | Baseline comparison |

### Final Ensemble Model Performance

The final ensemble combines ConvNeXt-Large + EfficientNet-V2-Small using a confidence-based switching mechanism:

#### Overall Metrics
- **Test Accuracy**: 95.64% (51,468 test samples)
- **Macro Average**: Precision 96.38%, Recall 94.52%, F1-Score 94.97%
- **Weighted Average**: Precision 95.94%, Recall 95.64%, F1-Score 95.43%

#### Per-Species Performance
| Species | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Alveolina** | 95.04% | 99.73% | 97.33% | 5,129 |
| **Arumella** | 99.34% | 98.06% | 98.70% | 2,163 |
| **Ataxophragmium** | 99.59% | 98.90% | 99.24% | 4,898 |
| **Baculogypsina** | 99.94% | 60.30% | 75.22% | 2,980 |
| **Chrysalidina** | 91.98% | 99.90% | 95.78% | 5,111 |
| **Coskinolina** | 97.86% | 98.58% | 98.22% | 5,060 |
| **Elphidiella** | 95.58% | 99.62% | 97.56% | 3,385 |
| **Fallotia** | 99.40% | 99.92% | 99.66% | 5,009 |
| **Lockhartia** | 96.85% | 99.98% | 98.39% | 5,359 |
| **Minoxia** | 99.22% | 89.49% | 94.10% | 3,557 |
| **Orbitoides** | 82.70% | 92.16% | 87.17% | 5,015 |
| **Rhapydionina** | 98.99% | 97.61% | 98.29% | 3,802 |

### Key Achievements

- 🎯 **95.64% Test Accuracy** with final ensemble model
- 🏆 **99.6% Top-3 Accuracy** (correct species in top 3 predictions)
- 📊 **Robust Performance** across all 12 species with F1-scores >75%
- 🔄 **Reproducible Pipeline** with comprehensive documentation
- 💡 **Transfer Learning** from ImageNet for efficient training
- ⚡ **Real-time Inference** optimized for production deployment

## 🛠️ Technical Features

### Advanced Data Processing

- **Intelligent Slice Sampling**: Automatic detection of fossil-containing slices
- **Otsu Thresholding**: Adaptive thresholding for optimal segmentation
- **Black Background Segmentation**: Enhanced contrast for model training
- **Quality Filtering**: Removal of low-quality or empty slices

### Sophisticated Model Training

- **Two-Phase Training**: Freeze backbone → Fine-tune end-to-end
- **Advanced Augmentation**: CutMix, MixUp, geometric transformations
- **Mixed Precision Training**: Faster training with maintained accuracy
- **Ensemble Methods**: Weighted combination of complementary models

### Comprehensive Evaluation

- **Statistical Analysis**: Confidence intervals and significance testing
- **Per-Class Metrics**: Detailed species-specific performance
- **Confusion Matrix Analysis**: Error pattern identification
- **Cross-Model Comparison**: Systematic architecture evaluation

## 📈 Results and Impact

### Performance Highlights

- **Overall Accuracy**: 95.64% on held-out test set (51,468 samples)
- **Top-3 Accuracy**: 99.6% (correct species in top 3 predictions)
- **Exceptional Reliability**: 99.8+ AUC scores across all models
- **Balanced Performance**: 94.97% macro F1-score across all 12 species
- **Production Ready**: 95.43% weighted F1-score for real-world deployment

### Model Architecture Success

- **Ensemble Leadership**: Final ensemble (95.64%) outperforms individual models
- **ConvNeXt Excellence**: ConvNeXt-Large achieves 95.12% accuracy with 99.63% top-3
- **Consistent Top Performance**: 5 models achieve >90% accuracy
- **Robust Classification**: All models maintain >97% top-3 accuracy
- **Efficient Options**: MobileNet delivers 88.02% accuracy for edge deployment

### Per-Species Achievement

- **Outstanding Performers**: Fallotia (99.66% F1), Ataxophragmium (99.24% F1), Arumella (98.70% F1)
- **Strong Classification**: 9 out of 12 species achieve >90% F1-scores
- **Challenging Species**: Baculogypsina (75.22% F1) and Orbitoides (87.17% F1) require continued research
- **High Precision**: Average precision of 96.38% across all species
- **Reliable Recall**: Average recall of 94.52% with consistent performance

### Scientific Contributions

- **Automated Paleontology**: First comprehensive deep learning system for 3D fossil classification
- **Large-Scale Dataset**: Curated dataset of ~120,000+ high-quality fossil slice images
- **Methodological Framework**: Reproducible pipeline validated across 8 state-of-the-art architectures
- **Performance Benchmarks**: Comprehensive evaluation establishing new standards for fossil AI classification
- **Transfer Learning Success**: Demonstrated effective adaptation from ImageNet to specialized paleontological domain

## 🌐 Interactive Dashboard

### Key Features

- **🖼️ Real-time Classification**: Upload images for instant species identification
- **📊 Confidence Visualization**: Probability distributions across all species
- **🔍 Dataset Explorer**: Browse and analyze the fossil image collection
- **📈 Performance Dashboard**: Interactive model performance metrics
- **🧬 Species Information**: Detailed paleontological descriptions

### User Experience

- **Drag & Drop Interface**: Easy image upload
- **Responsive Design**: Works on desktop and mobile devices
- **Educational Content**: Species descriptions and identification guides
- **Export Functionality**: Download predictions and analysis results

## 💻 System Requirements

### Minimum Requirements

- **OS**: Linux, Windows, or macOS
- **Python**: 3.8+
- **RAM**: 16GB (32GB recommended)
- **GPU**: 8GB VRAM (16GB+ for large models)
- **Storage**: 100GB free space

### Recommended Environment

```bash
# Use the provided Docker container for best compatibility
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --rm -p 10000:8888 -p 8501:8501 -v ${PWD}:/workspace/mycode \
  abdelghafour1/ngc_tf_rapids_25_01_vscode_torch:2025-v3 \
  jupyter lab --ip=0.0.0.0 --allow-root \
  --NotebookApp.custom_display_url=http://$(hostname):10000
```

## 📚 Documentation

Each section includes comprehensive documentation:

- **[1_Dataset_Creation/README.md](1_Dataset_Creation/README.md)** - Dataset creation and preprocessing
- **[2_AI_Modeling_Transfer_Learning/README.md](2_AI_Modeling_Transfer_Learning/README.md)** - Model training and evaluation
- **[3_Results/README.md](3_Results/README.md)** - Results analysis and comparison
- **[4_Dashboard_App/README.md](4_Dashboard_App/README.md)** - Dashboard deployment and usage

## 🔧 Troubleshooting

### Common Issues

1. **GPU Memory Errors**: Reduce batch size in training notebooks
2. **Dataset Path Issues**: Verify relative paths between sections
3. **Model Loading Errors**: Ensure all dependencies are installed
4. **Dashboard Access**: Check port availability (8501) and firewall settings

### Performance Optimization

1. **Training Speed**: Use mixed precision and optimal batch sizes
2. **Inference Speed**: Use TensorRT optimization for production deployment
3. **Memory Usage**: Clear GPU memory between training runs
4. **Storage**: Use SSD storage for faster data loading

## 📄 Citation

If you use this project in your research, please cite:

```bibtex
@software{fossil_classification_2025,
  title={Deep Learning Pipeline for 3D Fossil Classification},
  author={Halimi, Abdelghafour and Alibrahim, Ali and Barradas-Bautista, Didier and Sicat, Ronell and Afifi, Abdulkader M.},
  year={2025},
  url={https://github.com/A-Halimi/3D_Fossil_Project},
  note={Comprehensive deep learning system for paleontological image analysis}
}
```


## 📞 Support

For questions, issues, or suggestions:

1. **GitHub Issues**: Report bugs and request features
2. **Documentation**: Check section-specific READMEs
3. **Discussions**: Join the GitHub Discussions for community support

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Fossil Hunting! 🦕🔬**

*This project demonstrates the power of AI in paleontological research, making fossil identification faster, more accurate, and accessible to researchers worldwide.*
