# MLOps Repository - Complete Architecture & Blueprint

**Project**: California Housing Price Prediction - MLOps Pipeline  
**Repository**: https://github.com/sloumaaaaa/mlops.git  
**Branch**: dev (primary), main  
**Python Version**: 3.9+  
**Last Updated**: January 2026

---

## 📋 Table of Contents

1. [Complete Folder Structure](#complete-folder-structure)
2. [Key Files Overview](#key-files-overview)
3. [Configuration Files](#configuration-files)
4. [Script Files & Purposes](#script-files--purposes)
5. [MLOps Best Practices](#mlops-best-practices)
6. [Technology Stack](#technology-stack)
7. [Pipeline Architecture](#pipeline-architecture)

---

## 📁 Complete Folder Structure

```
mlops/
├── .github/
│   └── workflows/
│       └── ml_pipeline.yml              # GitHub Actions CI/CD configuration
├── .dvc/                                # DVC configuration directory
├── src/                                 # Core source code
│   ├── data_loader.py                  # Dataset loading & versioning (3 versions)
│   ├── preprocessing.py                # Data preprocessing utilities
│   ├── train.py                        # Model training with MLflow integration
│   ├── hyperparameter_tuning.py        # Optuna-based hyperparameter optimization
│   └── evaluate.py                     # Model evaluation & comparison
├── data/                                # Versioned datasets
│   ├── v1_california_housing.csv       # Original: 20,640 rows × 9 columns
│   ├── v2_filtered_housing.csv         # Filtered: 10,297 rows × 9 columns
│   └── v3_engineered_housing.csv       # Feature-engineered: 10,297 rows × 13 columns
├── models/                              # Trained model artifacts
│   ├── random_forest_v1.pkl
│   ├── xgboost_v2.pkl
│   ├── xgboost_v3.pkl
│   └── xgboost_optimized_v3.pkl        # Best model after hyperparameter tuning
├── results/                             # Generated reports & visualizations
│   ├── model_comparison.txt             # Comparison report
│   ├── model_comparison.png             # Performance visualization
│   ├── comparison_by_version.png        # Data version comparison
│   ├── comparison_by_model.png          # Model type comparison
│   └── optuna/
│       ├── xgboost_best_params.json
│       ├── xgboost_study.pkl
│       ├── xgboost_plots_history.png
│       └── xgboost_plots_importance.png
├── artifacts/                           # Model predictions & plots
│   ├── predictions_random_forest_v1.png
│   ├── predictions_xgboost_v2.png
│   └── predictions_xgboost_v3.png
├── .dvcignore                           # DVC ignore patterns
├── .gitignore                           # Git ignore patterns
├── dvc.yaml                             # DVC pipeline definition (8 stages)
├── dvc.lock                             # DVC lock file for reproducibility
├── requirements.txt                     # Python dependencies
├── run_complete_workflow.py             # Automated full workflow script
├── change_dataset.py                    # Dataset version management utility
├── train.py                             # Legacy training script (also in src/)
├── train_mlflow.py                      # MLflow-specific training
├── generate.py                          # Data generation utility
├── README.md                            # Project overview
├── DOCUMENTATION.md                     # Technical documentation
├── GUIDE_EXECUTION.md                   # Step-by-step execution guide
├── INSTALLATION.md                      # Installation instructions
├── QUICK_REFERENCE.md                   # Command reference
├── PROJET_RESUME.md                     # Project summary (French)
└── mlflow.db                            # MLflow tracking database (auto-generated)
```

---

## 📄 Key Files Overview

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Main project overview with quick start guide |
| `DOCUMENTATION.md` | Complete technical documentation (tools, architecture, results) |
| `INSTALLATION.md` | Step-by-step installation & first-time setup |
| `GUIDE_EXECUTION.md` | Detailed execution guide with expected outputs |
| `QUICK_REFERENCE.md` | Command reference card for common operations |
| `PROJET_RESUME.md` | French-language project summary |

### Configuration & Metadata

| File | Purpose |
|------|---------|
| `requirements.txt` | All Python dependencies with versions |
| `dvc.yaml` | DVC pipeline definition (8 stages) |
| `dvc.lock` | DVC reproducibility lock |
| `.gitignore` | Git exclusion patterns |
| `.dvcignore` | DVC exclusion patterns |
| `.github/workflows/ml_pipeline.yml` | GitHub Actions CI/CD workflow |

### Entry Point Scripts

| File | Purpose |
|------|---------|
| `run_complete_workflow.py` | Execute entire pipeline automatically |
| `change_dataset.py` | Manage & switch dataset versions |
| `generate.py` | Generate synthetic data |

---

## ⚙️ Configuration Files

### 1. **requirements.txt** - Dependency Management

```
# Core ML Libraries
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# MLOps Tools
mlflow>=2.9.0
dvc>=3.30.0

# Advanced ML Models
xgboost>=2.0.0
lightgbm>=4.1.0

# Hyperparameter Optimization
optuna>=3.4.0
optuna-integration>=3.4.0

# Data Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
kaleido>=0.2.1

# Utilities
pyyaml>=6.0
python-dotenv>=1.0.0
joblib>=1.3.0

# Testing & Code Quality
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.1.0
isort>=5.12.0
```

### 2. **dvc.yaml** - Pipeline Orchestration

8-stage DVC pipeline:

```yaml
stages:
  # Stages 1-3: Data Loading
  load_v1:
    cmd: python src/data_loader.py --version 1
    deps: [src/data_loader.py]
    outs: [data/v1_california_housing.csv]

  load_v2:
    cmd: python src/data_loader.py --version 2
    deps: [src/data_loader.py]
    outs: [data/v2_filtered_housing.csv]

  load_v3:
    cmd: python src/data_loader.py --version 3
    deps: [src/data_loader.py]
    outs: [data/v3_engineered_housing.csv]

  # Stages 4-6: Model Training
  train_v1:
    cmd: python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1
    deps: [src/train.py, data/v1_california_housing.csv]
    metrics: [artifacts/metrics_v1.json: {cache: false}]

  train_v2:
    cmd: python src/train.py --data_path data/v2_filtered_housing.csv --model xgboost --data_version v2
    deps: [src/train.py, data/v2_filtered_housing.csv]
    metrics: [artifacts/metrics_v2.json: {cache: false}]

  train_v3:
    cmd: python src/train.py --data_path data/v3_engineered_housing.csv --model xgboost --data_version v3
    deps: [src/train.py, data/v3_engineered_housing.csv]
    metrics: [artifacts/metrics_v3.json: {cache: false}]

  # Stage 7: Hyperparameter Tuning
  optimize:
    cmd: python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model xgboost --n_trials 50 --data_version v3
    deps: [src/hyperparameter_tuning.py, data/v3_engineered_housing.csv]
    outs: [results/optuna/xgboost_best_params.json, results/optuna/xgboost_study.pkl]

  # Stage 8: Model Evaluation
  evaluate:
    cmd: python src/evaluate.py --compare_all
    deps: [src/evaluate.py]
```

**Execution**: `dvc repro` (reproduces entire pipeline)

### 3. **.github/workflows/ml_pipeline.yml** - CI/CD Automation

**Triggers**:
- Push to `main` and `dev` branches
- Pull requests to `main`
- Manual trigger (`workflow_dispatch`)
- Scheduled: Monday 2 AM UTC

**6 Sequential Jobs**:

1. **code-quality** - Linting & formatting
   - Black (code formatter check)
   - isort (import sorting check)
   - flake8 (linter)

2. **data-validation** - Data integrity checks
   - Load all 3 dataset versions
   - Validate structure & content
   - Upload as artifacts (7-day retention)

3. **model-training** - Matrix strategy
   - 2 models × 3 versions = 6 parallel jobs
   - Models: random_forest, xgboost
   - Versions: v1, v2, v3
   - Save artifacts (30-day retention)

4. **hyperparameter-tuning** - Optuna optimization
   - 20 trials (CI-optimized)
   - 20-minute timeout
   - Save best params & study

5. **model-evaluation** - Comparison reports
   - Compare all trained models
   - Generate visualizations
   - Create comparison report

6. **summary-report** - Final summary
   - GitHub Step Summary
   - PR comment with results
   - Pipeline status overview

---

## 🐍 Script Files & Purposes

### 1. **src/data_loader.py** - Dataset Management

**Purpose**: Load and create 3 versions of California Housing dataset

**Key Functions**:
- `load_raw_data()` - Fetch dataset or generate synthetic data
- `create_version_1()` - Original dataset (20,640 rows)
- `create_version_2()` - Filtered dataset (outliers removed, coastal focus)
- `create_version_3()` - Feature-engineered dataset

**Features**:
- Fallback to synthetic data if HTTP 403 error
- Automatic directory creation
- Detailed logging

**Usage**:
```bash
python src/data_loader.py --version 1  # Or 2 or 3
python src/data_loader.py --version 2 --output_dir custom_path
```

**Dataset Versions**:

| Version | Size | Key Transformations |
|---------|------|-------------------|
| V1 | 20,640 × 9 | Original dataset |
| V2 | 10,297 × 9 | Remove outliers, coastal focus |
| V3 | 10,297 × 13 | +4 engineered features |

**V3 Engineered Features**:
- `rooms_per_household` = AveRooms × AveOccup
- `bedrooms_ratio` = AveBedrms / AveRooms
- `population_density` = Population / AveOccup
- `income_category_encoded` = categorical encoding of MedInc ranges

---

### 2. **src/train.py** - Model Training with MLflow

**Purpose**: Train multiple regression models with experiment tracking

**Class**: `ModelTrainer`
- Supports 4 models: random_forest, gradient_boosting, ridge, xgboost (optional)
- Automatic hyperparameter defaults
- Custom parameter override support

**Key Features**:
- MLflow integration (track params, metrics, artifacts)
- Prediction visualization (actual vs predicted + residuals)
- Model persistence with joblib
- Comprehensive logging

**Metrics Tracked**:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of determination)
- MAPE (Mean Absolute Percentage Error)
- Training time

**Usage**:
```bash
# Random Forest on V1
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1

# XGBoost on V3 with custom hyperparameters
python src/train.py --data_path data/v3_engineered_housing.csv --model xgboost --data_version v3 \
  --n_estimators 150 --max_depth 7 --learning_rate 0.05

# View results
python -m mlflow ui --port 5000
```

**Default Hyperparameters**:

| Model | n_estimators | max_depth | learning_rate |
|-------|--------------|-----------|---------------|
| Random Forest | 100 | 10 | - |
| Gradient Boosting | 100 | 5 | 0.1 |
| XGBoost | 100 | 6 | 0.1 |
| Ridge | - | - | alpha=1.0 |

---

### 3. **src/hyperparameter_tuning.py** - Optuna Optimization

**Purpose**: Automated hyperparameter optimization using Bayesian search

**Class**: `HyperparameterTuner`
- Supports: random_forest, gradient_boosting, xgboost
- Cross-validation integrated
- MLflow callback for experiment tracking

**Search Spaces** (examples):

**Random Forest**:
- n_estimators: 50-300
- max_depth: 3-20
- min_samples_split: 2-20
- min_samples_leaf: 1-10
- max_features: ['sqrt', 'log2', None]

**XGBoost**:
- n_estimators: 50-300
- max_depth: 3-10
- learning_rate: 0.01-0.3
- subsample: 0.6-1.0
- colsample_bytree: 0.6-1.0
- reg_alpha, reg_lambda: 0-1

**Usage**:
```bash
# 100 trials on V3 with XGBoost
python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv \
  --model xgboost --n_trials 100 --cv_folds 5 --data_version v3

# Output: Best params, study artifact, optimization plots
```

**Outputs**:
- `results/optuna/xgboost_best_params.json` - Best parameters
- `results/optuna/xgboost_study.pkl` - Optuna study object
- `results/optuna/xgboost_plots_history.png` - Optimization history
- `results/optuna/xgboost_plots_importance.png` - Parameter importance

---

### 4. **src/evaluate.py** - Model Comparison

**Purpose**: Compare models across different versions and data versions

**Class**: `ModelComparator`
- Fetches all runs from MLflow
- Generates comparison DataFrames
- Creates visualizations
- Generates comprehensive reports

**Comparison Functions**:
- `fetch_mlflow_runs()` - Get all experiment runs
- `create_comparison_dataframe()` - Tabular comparison
- `get_best_runs()` - Top N performers
- `compare_by_data_version()` - Performance across V1, V2, V3
- `compare_by_model_type()` - Model comparison
- `generate_report()` - Text report generation

**Usage**:
```bash
# Generate all comparisons
python src/evaluate.py --compare_all

# Custom experiment comparison
python src/evaluate.py --experiments california-housing optuna-tuning

# Output: Comparison plots, statistics, reports
```

**Outputs**:
- `results/comparison_by_version.png` - Box plots by data version
- `results/comparison_by_model.png` - RMSE & R² by model
- `results/comparison_report.txt` - Statistical summary

---

### 5. **run_complete_workflow.py** - Orchestration Script

**Purpose**: Execute entire pipeline end-to-end

**Execution Steps**:
1. Create all 3 dataset versions
2. Train baseline models
3. Optimize hyperparameters
4. Evaluate and compare
5. Generate reports

**Features**:
- Progress tracking
- Error handling
- Automatic logging
- Summary statistics

**Usage**:
```bash
python run_complete_workflow.py
```

---

### 6. **src/preprocessing.py** - Data Preprocessing

**Purpose**: Data cleaning and transformation utilities

**Typical Operations**:
- Outlier removal
- Feature scaling
- Handling missing values
- Feature engineering

---

## 🎯 MLOps Best Practices Implemented

### 1. **Data Versioning (DVC)**
- ✅ 3 dataset versions with different transformations
- ✅ DVC pipeline for reproducible data flow
- ✅ Metadata tracking with dvc.lock
- ✅ Branching support for different data strategies

### 2. **Experiment Tracking (MLflow)**
- ✅ Centralized experiment tracking
- ✅ Parameter logging (hyperparameters, data paths)
- ✅ Metric tracking (RMSE, MAE, R², MAPE)
- ✅ Artifact storage (models, plots, visualizations)
- ✅ Run comparison and best model identification
- ✅ SQLite backend for persistence

### 3. **Model Development**
- ✅ Multiple model implementations (RandomForest, GradientBoosting, XGBoost, Ridge)
- ✅ Hyperparameter optimization (Optuna with Bayesian search)
- ✅ Cross-validation for robust evaluation
- ✅ Model serialization (joblib)
- ✅ Prediction visualization

### 4. **Code Quality**
- ✅ Black formatting enforcement
- ✅ isort import organization
- ✅ flake8 linting
- ✅ Comprehensive logging
- ✅ Modular code structure
- ✅ Type hints in function signatures

### 5. **CI/CD Automation (GitHub Actions)**
- ✅ Multi-job workflow with dependencies
- ✅ Matrix testing (2 models × 3 versions)
- ✅ Artifact caching and retention policies
- ✅ Automated code quality checks
- ✅ Scheduled runs (weekly)
- ✅ Manual workflow dispatch
- ✅ PR comments with results
- ✅ GitHub Step Summary

### 6. **Documentation**
- ✅ README with quick start
- ✅ Technical documentation
- ✅ Installation guide
- ✅ Execution guide
- ✅ Quick reference commands
- ✅ Inline code comments

### 7. **Reproducibility**
- ✅ Fixed random seeds (42)
- ✅ Dependency pinning
- ✅ DVC lock file
- ✅ Environment specification
- ✅ Data versioning

---

## 🛠️ Technology Stack

### Core ML Libraries
- **scikit-learn** (1.3+) - ML models & preprocessing
- **XGBoost** (2.0+) - Gradient boosting
- **LightGBM** (4.1+) - Fast boosting alternative
- **NumPy** (1.24+) - Numerical computing
- **Pandas** (2.0+) - Data manipulation

### MLOps Tools
- **MLflow** (2.9+) - Experiment tracking & model registry
- **DVC** (3.30+) - Data & model versioning
- **Optuna** (3.4+) - Hyperparameter optimization

### Visualization
- **Matplotlib** (3.7+) - Static plots
- **Seaborn** (0.12+) - Statistical plots
- **Plotly** (5.17+) - Interactive plots
- **Kaleido** (0.2+) - Plot export to PNG/SVG

### DevOps & CI/CD
- **GitHub Actions** - CI/CD pipeline
- **Git** - Version control
- **pytest** - Testing framework

### Code Quality
- **Black** - Code formatter
- **isort** - Import sorter
- **flake8** - Linter

---

## 🏗️ Pipeline Architecture

### Data Flow

```
Raw Data (Synthetic/Fetched)
    ↓
Version 1 (Original) → Train RandomForest → Evaluate
    ↓
Version 2 (Filtered) → Train XGBoost → Evaluate
    ↓
Version 3 (Engineered) → Train XGBoost → Hyperparameter Tuning → Evaluate
    ↓
Model Comparison & Report Generation
```

### Training Workflow

```
Load Data
    ↓
Train/Validation Split (80/20)
    ↓
Model Training
    ↓
Predictions & Visualization
    ↓
Metrics Calculation
    ↓
MLflow Logging (params, metrics, artifacts)
    ↓
Model Serialization
```

### Hyperparameter Optimization Loop

```
Initialize Optuna Study
    ↓
For each trial (n_trials):
    ├── Sample hyperparameters
    ├── Cross-validation (k_folds)
    ├── Calculate CV RMSE
    ├── Log to MLflow
    ↓
Return best parameters
    ↓
Retrain on full training set
    ↓
Final evaluation on test set
```

---

## 📊 Results Summary

### Performance Metrics

| Model | Data Version | RMSE | R² | MAE | MAPE |
|-------|--------------|------|-----|-----|------|
| RandomForest | V1 | 0.4059 | 0.9031 | 0.2925 | 11.72% |
| XGBoost | V2 | 0.4043 | 0.9060 | 0.2908 | 11.62% |
| XGBoost | V3 | 0.4023 | 0.9069 | 0.2894 | 11.55% |
| **XGBoost (Optimized)** | **V3** | **0.3980** | **0.9095** | **0.2870** | **11.40%** |

### Key Improvements
- **V1 → V3**: 0.88% improvement in RMSE
- **Post-Tuning**: Additional 1.07% improvement
- **Data Filtering**: 50% data reduction (20,640 → 10,297) maintains performance
- **Feature Engineering**: Consistent improvements across metrics

---

## 🚀 Quick Start Commands

```bash
# Clone & Setup
git clone https://github.com/sloumaaaaa/mlops.git
cd mlops
python -m venv venv
./venv/Scripts/Activate  # or source venv/bin/activate

# Install
pip install -r requirements.txt

# Create Datasets
python src/data_loader.py --version 1
python src/data_loader.py --version 2
python src/data_loader.py --version 3

# Train Models
python src/train.py --data_path data/v1_california_housing.csv --model random_forest --data_version v1
python src/train.py --data_path data/v3_engineered_housing.csv --model xgboost --data_version v3

# Hyperparameter Tuning
python src/hyperparameter_tuning.py --data_path data/v3_engineered_housing.csv --model xgboost --n_trials 50

# Evaluate
python src/evaluate.py --compare_all

# View Results
python -m mlflow ui --port 5000
# Open http://127.0.0.1:5000

# Or run everything
python run_complete_workflow.py

# DVC Pipeline
dvc repro
dvc metrics show
```

---

## 🎓 Blueprint for Similar MLOps Projects

To replicate this structure for your own ML project:

### 1. **Directory Structure** (Essential)
```
project/
├── src/                  # All Python code
├── data/                 # Versioned datasets
├── models/               # Trained models
├── results/              # Reports & visualizations
├── .github/workflows/    # CI/CD configuration
├── requirements.txt      # Dependencies
├── dvc.yaml             # Pipeline definition
└── README.md            # Documentation
```

### 2. **Core Files** (Required)
- `src/data_loader.py` - Data management
- `src/train.py` - Training with MLflow
- `src/evaluate.py` - Model comparison
- `requirements.txt` - Dependencies
- `.gitignore` & `.dvcignore`

### 3. **Configuration Files**
- `dvc.yaml` - 8-stage pipeline
- `.github/workflows/ml_pipeline.yml` - 6-job CI/CD

### 4. **Documentation** (Recommended)
- README.md
- DOCUMENTATION.md
- INSTALLATION.md
- QUICK_REFERENCE.md

### 5. **Key Practices**
- Use MLflow for experiment tracking
- Use DVC for data versioning
- Implement multiple dataset versions
- Automate with GitHub Actions
- Use Optuna for hyperparameter tuning
- Maintain comprehensive logging
- Version control everything (except data artifacts)

---

## 📝 Notes for Implementation

1. **Data Privacy**: The project uses synthetic data to avoid privacy concerns
2. **Modularity**: Each component is independent and can be used separately
3. **Scalability**: Pipeline can be extended with additional models or data versions
4. **Monitoring**: MLflow provides complete experiment history
5. **Reproducibility**: Fixed seeds and DVC ensure reproducible results
6. **Extensibility**: Easy to add new models, metrics, or data transformations

---

**Project Created**: ESPRIT - January 2026  
**Contributors**: mhached-ai, AmineHached, MohamedYassineKhaldi00  
**License**: Open Source  
**Language**: 100% Python
