# Diabetic-Retinopathy-Exudate-Detection
Project Overview
This project provides a comprehensive workflow for detecting exudates—key retinal lesions associated with diabetic retinopathy—using retinal fundus images from publicly available clinical datasets (APTOS and IDRiD). The goal is to develop a robust, automated system that supports early screening and diagnosis of diabetic retinopathy with high accuracy, reliability, and scalability to real-world clinical settings.

The system includes advanced preprocessing steps such as automated optic disc removal, contrast enhancement, and green channel emphasis, combined with a multi-scale feature fusion approach using a Feature Pyramid Network (FPN) built on the EfficientNetB0 backbone. The pipeline is designed for batch processing, extensive error handling, and clinical workflow integration.

Features
    Automated optic disc localization and removal using intensity and anatomical prior methods
    Image enhancement via CLAHE and green channel normalization
    Multi-scale feature extraction and fusion with FPN and EfficientNetB0 backbone
    Binary classification of retinal images based on presence or absence of exudates
    Support for large-scale batch processing of fundus image datasets
    Performance evaluation with sensitivity, specificity, accuracy, and F1-score metrics
    Visualization tools for segmentation masks and overlay results
    Modular and configurable pipeline enabling reproducible experiments and parameter tuning

Dataset
  APTOS: A large dataset of retinal fundus images used for diabetic retinopathy screening
  IDRiD: A retinal image dataset with detailed annotations for lesions including exudates

Requirements
  Python 3.8 or higher
  OpenCV
  Pillow
  NumPy
  scikit-image
  scikit-learn
  PyTorch 
  matplotlib
  seaborn

imgaug or albumentations (for data augmentation)

Additional libraries for logging, configuration handling, and experiment tracking (optional)
