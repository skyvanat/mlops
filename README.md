# 🏅 Olympic Medals Prediction - MLOps Pipeline

## 📋 Project Overview

Complete MLOps pipeline for predicting Olympic medal winners using Machine Learning. This project demonstrates industrial-grade ML practices including:

- **Data Versioning** with DVC (3 dataset versions)
- **Experiment Tracking** with MLflow
- **Automated Training Pipeline** with DVC stages
- **CI/CD Automation** with GitHub Actions
- **Advanced Feature Engineering** with automated class weight balancing
- **Model Reproducibility** with version control

## 📊 Dataset

**Source**: 125 Years of Summer Olympics (Kaggle)

- **Total Records**: 1,300+ country-level Olympic records
- **Time Period**: 1932-2016
- **Target Variable**: Medal (Gold, Silver, Bronze, None)
- **Features**: Year, Country, Host City, Historical Statistics, etc.

## 🏗️ Project Structure

```
.
├── data/
│   ├── raw/                      # Original data
│   │   └── olympics_raw.csv
│   └── processed/                # Processed data versions
│       ├── olympics_cleaned.csv
│       └── olympics_featured.csv
├── src/
│   ├── data_preparation.py       # Stage 1: Data cleaning
│   ├── feature_engineering.py    # Stage 2: Feature engineering
│   ├── train.py                  # Stage 3: Model training
│   └── utils.py                  # Utility functions
├── models/
│   ├── random_forest_model.pkl   # Trained model
│   └── metrics.json              # Performance metrics
├── .github/workflows/
│   └── ci_pipeline.yml           # GitHub Actions workflow
├── params.yaml                   # Configuration parameters
├── dvc.yaml                      # DVC pipeline definition
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize DVC (Optional)

```bash
dvc init
git add .dvc
git commit -m "Initialize DVC"
```

### 3. Run Complete Pipeline

```bash
# Method 1: Using DVC
dvc repro

# Method 2: Manual execution
python src/data_preparation.py
python src/feature_engineering.py
python src/train.py
```

### 4. View MLflow Results

```bash
mlflow ui
```

Open browser to `http://localhost:5000`

## 📈 Pipeline Stages

### Stage 1: Data Preparation (`data_preparation.py`)

**Input**: `data/raw/olympics_raw.csv`
**Output**: `data/processed/olympics_cleaned.csv`

Operations:
- Remove duplicates
- Handle missing values
- Normalize formats
- Create target variable (Medal)

### Stage 2: Feature Engineering (`feature_engineering.py`)

**Input**: `data/processed/olympics_cleaned.csv`
**Output**: `data/processed/olympics_featured.csv`

New Features:
- Historical statistics per country (total medals, ratios)
- Yearly statistics (total medals per year)
- Host advantage flag (country == host)
- Medal strength (normalized medal count)
- Encoded categorical variables

### Stage 3: Model Training (`train.py`)

**Input**: `data/processed/olympics_featured.csv`
**Output**: `models/random_forest_model.pkl`, `models/metrics.json`

Features:
- **Model**: Random Forest Classifier (100 trees, max_depth=5)
- **Class Balancing**: Automatic detection and balanced class weights
- **Metrics**: Accuracy, F1-Score (weighted)
- **MLflow Tracking**: All hyperparameters and metrics logged
- **Feature Importance**: Top 10 features reported

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | ~68-72% |
| Test F1-Score | ~0.65-0.70 |
| Train Accuracy | ~75-80% |

*Varies based on data version and feature engineering*

## 🔧 Configuration

Edit `params.yaml` to modify:

```yaml
train:
  random_state: 42
  test_size: 0.2
  
model:
  n_estimators: 100      # Number of trees
  max_depth: 5           # Tree depth limit
  class_weight: balanced # Handle class imbalance
```

## 📊 Advanced Features

### 1. Automatic Class Imbalance Handling

The training pipeline automatically:
- Detects imbalanced medal classes
- Calculates balanced class weights
- Applies weights during model training

This ensures fair prediction across all medal types including the majority "None" class.

### 2. Stratified Train-Test Split

Data is split maintaining medal distribution to avoid class imbalance in train/test sets.

### 3. Multi-Level Feature Importance

Reports top 10 features by importance to understand model decisions.

## 🔄 Experiment Tracking with MLflow

All experiments are automatically tracked:

```bash
mlflow ui
```

View:
- Hyperparameters for each run
- Metrics (Accuracy, F1-Score)
- Model artifacts
- Run duration and parameters

## 🤖 CI/CD Pipeline

GitHub Actions automatically runs on every push:

```yaml
jobs:
  1. Setup environment
  2. Install dependencies
  3. Run data preparation
  4. Run feature engineering
  5. Train model
  6. Log metrics to MLflow
```

## 📝 Data Versions

### Version 1: Raw Data
- Original format from Kaggle
- File: `olympics_raw.csv`

### Version 2: Cleaned Data
- Duplicates removed
- Missing values handled
- Target variable created
- File: `olympics_cleaned.csv`

### Version 3: Featured Data
- Advanced features engineered
- Categorical variables encoded
- Historical statistics added
- File: `olympics_featured.csv`

## 🔍 Performance Comparison

Impact of feature engineering:

| Version | Accuracy | F1-Score | Improvement |
|---------|----------|----------|------------|
| Raw | ~65% | 0.60 | Baseline |
| Cleaned | ~68% | 0.63 | +3% |
| Featured | ~71% | 0.68 | +6% |

## 🛠️ Troubleshooting

### Issue: Missing data files

```bash
# Regenerate from raw data
dvc repro --force
```

### Issue: MLflow not found

```bash
pip install mlflow
```

### Issue: Git/DVC conflicts

```bash
git status
dvc status
```

## 🤝 Contributing

1. Create a new branch
2. Make changes
3. Run: `dvc repro` to validate
4. Push to GitHub
5. CI/CD pipeline will automatically test

## 📚 Technologies Used

- **ML Framework**: scikit-learn
- **Data Processing**: pandas, numpy
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC (Data Version Control)
- **Orchestration**: DVC Pipelines
- **CI/CD**: GitHub Actions
- **Version Control**: Git

## 📄 License

This project is for educational purposes.

## 🔗 References

- Dataset: https://www.kaggle.com/code/hamdallak/125-years-of-summer-olympics-analysis-visual/input
- DVC Documentation: https://dvc.org
- MLflow Documentation: https://mlflow.org
- scikit-learn: https://scikit-learn.org

---

**Created**: January 2026
**Last Updated**: January 6, 2026
