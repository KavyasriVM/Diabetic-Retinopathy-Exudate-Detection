# Diabetic-Retinopathy-Exudate-Detection

## Project Overview
This project provides a **comprehensive automated workflow** for detecting **exudates**—key retinal lesions linked to **diabetic retinopathy**—using retinal fundus images from publicly available clinical datasets (**APTOS** and **IDRiD**).  
The goal is to develop a **robust, scalable, and clinically viable** screening system that ensures **early detection** and **accurate diagnosis**.

The system integrates:
- **Advanced preprocessing** (optic disc removal, contrast enhancement, green channel emphasis)
- **Multi-scale feature fusion** using a **Feature Pyramid Network (FPN)** with an **EfficientNetB0** backbone
- **Automated batch processing** for large-scale datasets
- **Clinical workflow integration** and reproducible experiments

---

## Features
- **Automated Optic Disc Localization & Removal**  
  - Uses intensity thresholding and anatomical priors
  - Reduces false positives in exudate detection
- **Image Enhancement**  
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)  
  - Green channel normalization for improved lesion visibility
- **Multi-scale Feature Extraction**  
  - FPN + EfficientNetB0 backbone
  - Enhanced detection of small and large lesions
- **Binary Classification**  
  - Presence vs. absence of exudates
- **Scalable Batch Processing**  
  - Handles thousands of images efficiently
- **Performance Metrics**  
  - Sensitivity, Specificity, Accuracy, F1-score
- **Visualization Tools**  
  - Segmentation masks  
  - Lesion overlay on fundus images
- **Modular & Configurable Pipeline**  
  - Easy parameter tuning and reproducibility

---

## Dataset
- **APTOS** – Large-scale dataset for diabetic retinopathy screening  
- **IDRiD** – Retinal images with detailed lesion annotations, including exudates

---

## Requirements
- **Python** ≥ 3.8  
- **Core Libraries**
  - `OpenCV`
  - `Pillow`
  - `NumPy`
  - `scikit-image`
  - `scikit-learn`
  - `PyTorch`
  - `matplotlib`
  - `seaborn`
- **Data Augmentation**
  - `imgaug` or `albumentations`
- **Optional**
  - Logging & experiment tracking libraries (`wandb`, `tensorboard`, `hydra-core`)

Install dependencies:
```bash
pip install -r requirements.txt
