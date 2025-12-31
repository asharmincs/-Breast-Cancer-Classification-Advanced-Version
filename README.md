# -Breast-Cancer-Classification-Advanced-Version
AI-driven breast cancer classification using deep learning on CBIS-DDSM mammogram images. Patient-wise train/val/test split, VGG16-based model, data preprocessing, and evaluation pipeline for accurate and reproducible results.

## Overview
Deep learning pipeline to classify breast cancer from mammogram images using CBIS-DDSM dataset. Includes data preprocessing, patient-wise train/validation/test split, VGG16-based model, and evaluation.

## Features
- Deep learning using VGG16
- Patient-wise split
- Train/Val/Test workflow
- Subset for quick experimentation
- Ready for scaling to full dataset

## Dataset
[CBIS-DDSM Breast Cancer Image Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

## Project Structure
breast-cancer-classification-advanced/
│
├── data/                      # (not uploaded to GitHub, ignored in .gitignore)
│   ├── train/
│   ├── val/
│   └── test/
│
├── notebooks/
│   └── breast_cancer_classification.ipynb
│
├── scripts/
│   ├── data_preparation.py
│   └── model_training.py
│
├── .gitignore
├── README.md
└── requirements.txt
