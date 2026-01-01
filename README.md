# Breast Cancer Classification using Deep Learning

This project presents an **AI-driven deep learning pipeline** for breast cancer classification using mammogram images from the **CBIS-DDSM dataset**. The focus is on **accuracy, reproducibility, and clinical relevance**, following patient-wise data splitting to avoid data leakage.

## Project Highlights
- CBIS-DDSM mammogram dataset
- Patient-wise Train / Validation / Test split
- Deep learning with transfer learning (VGG16)
- Biomedical image preprocessing
- Reproducible research workflow
- Evaluation on unseen test data

## Objective
To build a robust deep learning model that classifies mammogram images as **Benign** or **Malignant**, supporting research in **AI-assisted cancer diagnostics**.

## Dataset
- **Source:** CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
- **Modality:** Mammography
- **Classes:** Benign, Malignant
- **Access:** Kaggle

Dataset link:  
https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

> Dataset is not included in this repository due to size constraints.

## Project Structure

breast-cancer-classification-advance-version/
│
├── notebooks/
│ └── breast_cancer_classification.ipynb
│
├── scripts/
│ ├── dataset_preparation.py
│ ├── train_model.py
│ └── evaluate_model.py
│
├── models/
│ └── breast_cancer_model.keras
│
├── requirements.txt
├── README.md
└── LICENSE

## Tech Stack
- **Programming:** Python
- **Deep Learning:** TensorFlow / Keras
- **Model:** VGG16 (Transfer Learning)
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Platform:** Google Colab / Local

## How to Run (Google Colab Recommended)

 Install dependencies
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn kaggle
## Download dataset via Kaggle API
kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset
unzip cbis-ddsm-breast-cancer-image-dataset.zip

## Prepare dataset and train model Run the notebook :
notebooks/breast_cancer_classification_advance_version.ipynb

## Model Performance
Binary classification (Benign vs Malignant)
Evaluated using accuracy and loss
Patient-wise evaluation to ensure generalization

## Future Improvements
Vision Transformers (ViT)
Explainable AI (Grad-CAM)
Multi-view mammogram fusion
Model deployment (TensorFlow Serving)

# License (This project is licensed under the MIT License).
requirements.txt 

```txt
tensorflow
numpy
pandas
matplotlib
scikit-learn
kaggle

## Author
Sharmin Akhter
Machine Learning & Deep Learning Researcher
🔗 LinkedIn: https://linkedin.com/in/asharmincs
📚 Google Scholar: https://scholar.google.com/citations?user=rHqGuCoAAAAJ
📧 Email: asharmin.cs@gmail.com
