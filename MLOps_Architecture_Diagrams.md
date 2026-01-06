# MLOps Pipeline Architecture - Visual Guide

## 1. HIGH-LEVEL PIPELINE FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│                    MLOps Pipeline Architecture                   │
└─────────────────────────────────────────────────────────────────┘

INPUT LAYER (Data Sources)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Data Loading (src/data_loader.py)                              │
│ - Fetch California Housing or generate synthetic data           │
│ - Create 3 versions with different preprocessing                │
└─────────────────────────────────────────────────────────────────┘
    ↓
    ├─→ V1: Original (20,640 × 9)
    ├─→ V2: Filtered (10,297 × 9)
    └─→ V3: Engineered (10,297 × 13)
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Model Training (src/train.py + MLflow)                         │
│ - Load & split data (80/20)                                     │
│ - Train with 4 model types                                      │
│ - Log metrics, params, artifacts to MLflow                      │
│ - Save models with joblib                                       │
└─────────────────────────────────────────────────────────────────┘
    ↓
    ├─→ RandomForest on V1  → RMSE: 0.4059
    ├─→ XGBoost on V2       → RMSE: 0.4043
    └─→ XGBoost on V3       → RMSE: 0.4023
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Hyperparameter Tuning (src/hyperparameter_tuning.py + Optuna)  │
│ - Bayesian optimization over 50-100 trials                      │
│ - Cross-validation for robustness                               │
│ - Log best params to MLflow                                     │
│ - Retrain & evaluate final model                                │
└─────────────────────────────────────────────────────────────────┘
    ↓
    └─→ XGBoost Optimized on V3 → RMSE: 0.3980
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Model Evaluation (src/evaluate.py)                              │
│ - Fetch all runs from MLflow                                    │
│ - Compare across models & versions                              │
│ - Generate visualizations                                       │
│ - Create comprehensive reports                                  │
└─────────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ Output Layer (Results & Artifacts)                             │
│ - Trained models (.pkl files)                                   │
│ - Comparison plots (.png files)                                 │
│ - Performance reports (.txt files)                              │
│ - MLflow tracking database                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. DATA VERSIONING PIPELINE (DVC)

```
┌──────────────────────────────────────────────────────────────────┐
│                  dvc.yaml (8 Stages)                             │
└──────────────────────────────────────────────────────────────────┘

Stage 1: load_v1
├─ Input: src/data_loader.py
├─ Output: data/v1_california_housing.csv (20,640 rows)
└─ Command: python src/data_loader.py --version 1
    ↓
Stage 2: load_v2
├─ Input: src/data_loader.py
├─ Output: data/v2_filtered_housing.csv (10,297 rows)
└─ Command: python src/data_loader.py --version 2
    ↓
Stage 3: load_v3
├─ Input: src/data_loader.py
├─ Output: data/v3_engineered_housing.csv (10,297 rows × 13 cols)
└─ Command: python src/data_loader.py --version 3
    ↓
    ├─→ Stage 4: train_v1
    │  ├─ Model: RandomForest
    │  ├─ Metrics: artifacts/metrics_v1.json
    │  └─ Cmd: python src/train.py ... --model random_forest --data_version v1
    │
    ├─→ Stage 5: train_v2
    │  ├─ Model: XGBoost
    │  ├─ Metrics: artifacts/metrics_v2.json
    │  └─ Cmd: python src/train.py ... --model xgboost --data_version v2
    │
    └─→ Stage 6: train_v3
       ├─ Model: XGBoost
       ├─ Metrics: artifacts/metrics_v3.json
       └─ Cmd: python src/train.py ... --model xgboost --data_version v3
    ↓
Stage 7: optimize
├─ Input: src/hyperparameter_tuning.py, V3 data
├─ Outputs: results/optuna/xgboost_best_params.json
│           results/optuna/xgboost_study.pkl
└─ Cmd: python src/hyperparameter_tuning.py --n_trials 50 --model xgboost
    ↓
Stage 8: evaluate
├─ Input: src/evaluate.py
├─ Outputs: results/model_comparison.png
│           results/comparison_report.txt
└─ Cmd: python src/evaluate.py --compare_all

Execution: dvc repro
```

---

## 3. CI/CD WORKFLOW (GitHub Actions)

```
┌──────────────────────────────────────────────────────────────────┐
│         .github/workflows/ml_pipeline.yml (6 Jobs)               │
└──────────────────────────────────────────────────────────────────┘

Triggers: push(main,dev), PR(main), schedule(Mon 2AM), manual

    ↓

┌─────────────────────┐
│  Job 1: Code Quality │  [Runs-on: ubuntu-latest]
└─────────────────────┘
  Actions:
  ├─ Checkout code
  ├─ Setup Python 3.9
  ├─ Install tools (black, isort, flake8)
  ├─ Black format check
  ├─ isort import check
  └─ flake8 linting

    ↓
    needs: code-quality

┌──────────────────────┐
│  Job 2: Data Validation │  [Runs-on: ubuntu-latest]
└──────────────────────┘
  Actions:
  ├─ Checkout code
  ├─ Setup Python 3.9
  ├─ Install dependencies
  ├─ Initialize DVC
  ├─ Load V1, V2, V3 datasets
  ├─ Validate structure
  └─ Upload artifacts (7-day retention)

    ↓
    needs: data-validation

┌────────────────────────────┐
│  Job 3: Model Training (Matrix) │  [Runs-on: ubuntu-latest]
└────────────────────────────┘
  Matrix: 2 models × 3 versions = 6 parallel jobs
  
  Combinations:
  ├─ random_forest + v1
  ├─ random_forest + v2
  ├─ random_forest + v3
  ├─ xgboost + v1
  ├─ xgboost + v2
  └─ xgboost + v3
  
  Actions:
  ├─ Checkout code
  ├─ Setup Python 3.9
  ├─ Install dependencies
  ├─ Download datasets
  ├─ Train model
  └─ Upload models (30-day retention)

    ↓
    needs: data-validation (parallel)

┌────────────────────────────┐
│ Job 4: Hyperparameter Tuning │  [Runs-on: ubuntu-latest]
└────────────────────────────┘
  Actions:
  ├─ Checkout code
  ├─ Setup Python 3.9
  ├─ Install dependencies
  ├─ Download V3 dataset
  ├─ Run Optuna (20 trials, 20-min timeout)
  └─ Upload results (30-day retention)

    ↓
    needs: data-validation (parallel)

┌───────────────────────────┐
│ Job 5: Model Evaluation    │  [Runs-on: ubuntu-latest]
└───────────────────────────┘
  Actions:
  ├─ Checkout code
  ├─ Setup Python 3.9
  ├─ Install dependencies
  ├─ Generate comparison report
  └─ Upload results (30-day retention)

    ↓
    needs: model-training

┌────────────────────────┐
│ Job 6: Summary Report  │  [Runs-on: ubuntu-latest]
└────────────────────────┘
  Actions:
  ├─ Create GitHub Step Summary
  ├─ Generate pipeline status
  ├─ List artifacts
  ├─ Post PR comment (if applicable)
  └─ Workflow complete

    ↓
    needs: [model-training, hyperparameter-tuning, model-evaluation]

Result: All artifacts stored, PR commented, workflow summary generated
```

---

## 4. DATASET TRANSFORMATION PIPELINE

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dataset Evolution                             │
└─────────────────────────────────────────────────────────────────┘

Raw California Housing
(Fetched or Synthetic)
    │
    ├─→ 20,640 samples
    ├─→ 8 features (MedInc, HouseAge, AveRooms, AveBedrms, 
    │              Population, AveOccup, Latitude, Longitude)
    └─→ Target: MedHouseVal
    
    ↓ [create_version_1()]
    
VERSION 1 (Original)
    ├─ Size: 20,640 × 9 columns
    ├─ No transformations
    ├─ File: v1_california_housing.csv
    └─ RMSE (RandomForest): 0.4059
    
    ├─ AND ↓ [create_version_2()]
    
VERSION 2 (Filtered)
    ├─ Size: 10,297 × 9 columns (49.9% retained)
    ├─ Transformations:
    │  ├─ Remove outliers: MedHouseVal > 5.0 (eliminated high-value outliers)
    │  ├─ Geographic focus: Longitude < -119.0 (coastal regions only)
    │  └─ Drop NaN values
    ├─ File: v2_filtered_housing.csv
    └─ RMSE (XGBoost): 0.4043
    
    ├─ AND ↓ [create_version_3()]
    
VERSION 3 (Feature-Engineered)
    ├─ Size: 10,297 × 13 columns (same data as V2 + 4 new features)
    ├─ Inherits V2 filters
    ├─ New Features Created:
    │  ├─ rooms_per_household = AveRooms × AveOccup
    │  ├─ bedrooms_ratio = AveBedrms / AveRooms
    │  ├─ population_density = Population / AveOccup
    │  └─ income_category_encoded = categorical encoding (Low/Med/High/VeryHigh)
    ├─ File: v3_engineered_housing.csv
    └─ RMSE (XGBoost): 0.4023
    
    ↓ [Final: XGBoost after Optuna Tuning]
    
VERSION 3 (Optimized)
    ├─ Same data as V3
    ├─ Hyperparameters optimized via Optuna
    ├─ Best params found after 50 trials
    └─ RMSE: 0.3980 (1.07% improvement over baseline)
```

**Performance Progression**:
```
V1 (Original)           → RMSE: 0.4059 (Baseline)
V2 (Filtered)           → RMSE: 0.4043 (-0.39% improvement)
V3 (Engineered)         → RMSE: 0.4023 (-0.88% vs V1, -0.49% vs V2)
V3 (Optimized)          → RMSE: 0.3980 (-1.95% vs V1, -1.07% vs V3)
```

---

## 5. MLflow EXPERIMENT TRACKING STRUCTURE

```
┌──────────────────────────────────────────────────────────────────┐
│                    MLflow Tracking Database                       │
│                    (mlflow.db - SQLite)                          │
└──────────────────────────────────────────────────────────────────┘

Experiment: "california-housing" (Default training runs)
├─ Run 1: RandomForest on V1
│  ├─ Parameters:
│  │  ├─ data_path: data/v1_california_housing.csv
│  │  ├─ data_version: v1
│  │  ├─ model_type: random_forest
│  │  ├─ n_estimators: 100
│  │  ├─ max_depth: 10
│  │  └─ n_features: 9
│  ├─ Metrics:
│  │  ├─ rmse: 0.4059
│  │  ├─ mae: 0.2925
│  │  ├─ r2: 0.9031
│  │  ├─ mape: 11.72%
│  │  └─ training_time_seconds: 2.34
│  └─ Artifacts:
│     ├─ predictions_random_forest_v1.png
│     ├─ models/random_forest_v1.pkl
│     └─ artifacts/metrics_v1.json
│
├─ Run 2: XGBoost on V2
│  ├─ Parameters:
│  │  ├─ data_path: data/v2_filtered_housing.csv
│  │  ├─ data_version: v2
│  │  ├─ model_type: xgboost
│  │  ├─ n_estimators: 100
│  │  ├─ max_depth: 6
│  │  └─ n_features: 9
│  ├─ Metrics:
│  │  ├─ rmse: 0.4043
│  │  ├─ mae: 0.2908
│  │  ├─ r2: 0.9060
│  │  ├─ mape: 11.62%
│  │  └─ training_time_seconds: 1.89
│  └─ Artifacts:
│     ├─ predictions_xgboost_v2.png
│     ├─ models/xgboost_v2.pkl
│     └─ artifacts/metrics_v2.json
│
├─ Run 3: XGBoost on V3
│  ├─ Parameters:
│  │  ├─ data_path: data/v3_engineered_housing.csv
│  │  ├─ data_version: v3
│  │  ├─ model_type: xgboost
│  │  ├─ n_estimators: 100
│  │  ├─ max_depth: 6
│  │  └─ n_features: 13 ← (Engineered features)
│  ├─ Metrics:
│  │  ├─ rmse: 0.4023
│  │  ├─ mae: 0.2894
│  │  ├─ r2: 0.9069
│  │  ├─ mape: 11.55%
│  │  └─ training_time_seconds: 2.12
│  └─ Artifacts:
│     ├─ predictions_xgboost_v3.png
│     ├─ models/xgboost_v3.pkl
│     └─ artifacts/metrics_v3.json
│
└─ [More runs as needed]

Experiment: "optuna-tuning" (Hyperparameter optimization)
└─ Trial Runs (automated by MLflowCallback)
   ├─ Trial 1: RMSE 0.4050
   ├─ Trial 2: RMSE 0.4035
   ├─ Trial 3: RMSE 0.4012 ← Best
   ├─ ... (50 trials)
   └─ Best Params: {n_estimators: 120, max_depth: 7, learning_rate: 0.08, ...}

UI Access: python -m mlflow ui --port 5000
           → http://localhost:5000
```

---

## 6. DIRECTORY STRUCTURE WITH DEPENDENCIES

```
mlops/
│
├── .github/                          ← Version Control
│   └── workflows/
│       └── ml_pipeline.yml           → Triggers: 6-job CI/CD
│
├── .dvc/                             ← DVC Configuration
│   ├── config
│   └── .gitignore
│
├── src/                              ← Source Code (Core)
│   ├── data_loader.py ─────────┐
│   │                           │
│   ├── preprocessing.py         ├─→ Used by data_loader.py
│   │                           │
│   ├── train.py ────────────────┼─→ Uses: data from data_loader
│   │                           │   Logs to: MLflow
│   │                           │
│   ├── hyperparameter_tuning.py ┼─→ Uses: train.py code patterns
│   │                           │   Optimizes: XGBoost/RandomForest
│   │                           │   Logs to: MLflow + Optuna
│   │
│   └── evaluate.py ────────────────→ Reads: MLflow runs
│                                   Generates: Comparison reports
│
├── data/                             ← Versioned Datasets
│   ├── v1_california_housing.csv (20,640 × 9)    ← Version 1
│   ├── v2_filtered_housing.csv (10,297 × 9)      ← Version 2
│   └── v3_engineered_housing.csv (10,297 × 13)   ← Version 3
│                                                    (4 new features)
│
├── models/                           ← Trained Model Artifacts
│   ├── random_forest_v1.pkl
│   ├── xgboost_v2.pkl
│   ├── xgboost_v3.pkl
│   └── xgboost_optimized_v3.pkl      ← Best model after tuning
│
├── results/                          ← Reports & Visualizations
│   ├── model_comparison.txt
│   ├── model_comparison.png
│   ├── comparison_by_version.png
│   ├── comparison_by_model.png
│   └── optuna/
│       ├── xgboost_best_params.json
│       ├── xgboost_study.pkl
│       ├── xgboost_plots_history.png
│       └── xgboost_plots_importance.png
│
├── artifacts/                        ← Model Predictions
│   ├── predictions_random_forest_v1.png
│   ├── predictions_xgboost_v2.png
│   ├── predictions_xgboost_v3.png
│   └── metrics_*.json
│
├── Configuration Files
│   ├── requirements.txt               → pip dependencies
│   ├── dvc.yaml                      → 8-stage DVC pipeline
│   ├── dvc.lock                      → DVC reproducibility lock
│   ├── .gitignore                    → Git exclusions
│   └── .dvcignore                    → DVC exclusions
│
├── Orchestration Scripts
│   ├── run_complete_workflow.py       → End-to-end execution
│   ├── change_dataset.py              → Dataset version management
│   └── generate.py                    → Data generation
│
├── Documentation
│   ├── README.md
│   ├── DOCUMENTATION.md
│   ├── INSTALLATION.md
│   ├── GUIDE_EXECUTION.md
│   ├── QUICK_REFERENCE.md
│   └── PROJET_RESUME.md
│
└── Database
    └── mlflow.db                     ← MLflow tracking (auto-generated)

Legend:
─────→ Data Flow
→→→  → Execution Flow
```

---

## 7. MODEL TRAINING LIFECYCLE

```
┌─────────────────────────────────────────────────────────────────┐
│            Single Model Training Lifecycle                       │
│            (src/train.py + MLflow)                              │
└─────────────────────────────────────────────────────────────────┘

1. LOAD DATA
   └─ Read CSV → pandas DataFrame
      └─ Extract X (features) and y (target)
         └─ Train/Test Split (80/20)

2. INITIALIZE MLFLOW
   └─ Set experiment name (e.g., "california-housing")
      └─ Start run context
         └─ Log parameters:
            ├─ data_path
            ├─ data_version
            ├─ model_type
            ├─ n_estimators
            ├─ max_depth
            └─ [all hyperparameters]

3. INITIALIZE TRAINER
   └─ Create ModelTrainer instance
      └─ Select model class (RandomForest/XGBoost/etc)
         └─ Apply hyperparameters

4. TRAIN MODEL
   └─ model.fit(X_train, y_train)
      └─ Record training time

5. EVALUATE MODEL
   └─ predictions = model.predict(X_test)
      └─ Calculate metrics:
         ├─ RMSE = √(MSE)
         ├─ MAE = mean(|y - ŷ|)
         ├─ R² = 1 - (SSres/SStot)
         └─ MAPE = mean(|y - ŷ|/y) × 100%

6. VISUALIZE
   └─ Create 2-subplot figure:
      ├─ Subplot 1: Actual vs Predicted (scatter + perfect prediction line)
      └─ Subplot 2: Residuals (predictions vs residuals)
         └─ Save as PNG

7. LOG TO MLFLOW
   └─ mlflow.log_metrics()
      ├─ rmse: 0.4059
      ├─ mae: 0.2925
      ├─ r2: 0.9031
      └─ mape: 11.72%
   └─ mlflow.log_artifact()
      ├─ predictions_plot.png
      ├─ model.pkl
      └─ metrics.json
   └─ mlflow.sklearn.log_model()
      └─ Model registered in MLflow

8. SAVE MODEL
   └─ joblib.dump(model, 'models/xgboost_v3.pkl')

9. COMPLETE
   └─ Run tagged and available in MLflow UI
      └─ Can compare with other runs
```

---

## 8. HYPERPARAMETER OPTIMIZATION WORKFLOW (Optuna)

```
┌──────────────────────────────────────────────────────────────────┐
│         Hyperparameter Tuning Loop (src/hyperparameter_tuning.py)  │
│         Powered by Optuna (Bayesian Search)                       │
└──────────────────────────────────────────────────────────────────┘

START
  ↓
Load Data (data/v3_engineered_housing.csv)
  ↓
Train/Test Split
  ↓
Create Optuna Study
  ├─ Direction: minimize (objective = CV RMSE)
  └─ Study name: xgboost_optimization
  ↓
FOR trial = 1 TO n_trials (e.g., 50):
  │
  ├─ Sample Hyperparameters (Trial)
  │  ├─ n_estimators: int[50, 300]
  │  ├─ max_depth: int[3, 10]
  │  ├─ learning_rate: float[0.01, 0.3]
  │  ├─ min_child_weight: int[1, 7]
  │  ├─ subsample: float[0.6, 1.0]
  │  ├─ colsample_bytree: float[0.6, 1.0]
  │  ├─ gamma: float[0, 5]
  │  ├─ reg_alpha: float[0, 1]
  │  └─ reg_lambda: float[0, 1]
  │
  ├─ FOR fold = 1 TO cv_folds (e.g., 5):
  │  │
  │  ├─ Create model with sampled params
  │  ├─ Train on fold_train
  │  ├─ Evaluate on fold_test
  │  └─ Record RMSE
  │  
  ├─ Calculate mean CV RMSE across folds
  ├─ Log trial to MLflow (if callback enabled)
  ├─ Update study with this objective value
  ├─ Use Bayesian posterior to select next trial params
  │
  └─ If CV RMSE < best_rmse:
     └─ Update best_params and best_rmse
  
END FOR

↓
Output: best_params
  ├─ n_estimators: 120
  ├─ max_depth: 7
  ├─ learning_rate: 0.08
  ├─ subsample: 0.85
  └─ [other optimal params]

↓
Train Final Model
  ├─ Initialize XGBoost with best_params
  ├─ Train on full X_train
  ├─ Evaluate on X_test
  └─ Log final metrics to MLflow

↓
Save Results
  ├─ results/optuna/xgboost_best_params.json
  ├─ results/optuna/xgboost_study.pkl
  ├─ results/optuna/xgboost_plots_history.png (optimization progress)
  └─ results/optuna/xgboost_plots_importance.png (param importance)

↓
COMPLETE: Best model saved as models/xgboost_optimized_v3.pkl
```

---

## 9. DECISION TREE: WHICH SCRIPT TO USE

```
┌─ Start MLOps Pipeline
│
├─ Want to get started quickly?
│  └─ Run: python run_complete_workflow.py
│     (Executes everything end-to-end)
│
├─ Want to create/update datasets?
│  ├─ Create V1: python src/data_loader.py --version 1
│  ├─ Create V2: python src/data_loader.py --version 2
│  └─ Create V3: python src/data_loader.py --version 3
│
├─ Want to train a specific model?
│  └─ python src/train.py --data_path <path> --model <model> --data_version v1
│     (Logs to MLflow)
│
├─ Want to optimize hyperparameters?
│  └─ python src/hyperparameter_tuning.py --data_path <path> --model xgboost --n_trials 50
│     (Uses Optuna + MLflow)
│
├─ Want to compare models?
│  └─ python src/evaluate.py --compare_all
│     (Fetches MLflow runs, generates reports)
│
├─ Want to view results in UI?
│  └─ python -m mlflow ui --port 5000
│     (Opens http://localhost:5000)
│
├─ Want to run DVC pipeline?
│  ├─ Reproduce: dvc repro
│  ├─ View metrics: dvc metrics show
│  └─ Update artifacts: dvc push
│
└─ Want to use GitHub Actions?
   └─ Push to main/dev → Workflow auto-runs
      (6 jobs: code quality, data validation, training, tuning, eval, summary)
```

---

## 10. DATA SCHEMA

```
┌─────────────────────────────────────────────────────────────────┐
│                  California Housing Dataset Schema               │
└─────────────────────────────────────────────────────────────────┘

VERSION 1 (Original)
─────────────────────
Columns (9):
├─ MedInc (float64)       - Median income in block group
├─ HouseAge (float64)     - Median house age in block group
├─ AveRooms (float64)     - Average number of rooms per household
├─ AveBedrms (float64)    - Average number of bedrooms per household
├─ Population (float64)   - Block group population
├─ AveOccup (float64)     - Average house occupancy
├─ Latitude (float64)     - Block group latitude
├─ Longitude (float64)    - Block group longitude
└─ MedHouseVal (float64)  - TARGET: Median house value for CA districts

Shape: (20640, 9)
Target: MedHouseVal

VERSION 2 (Filtered)
────────────────────
Same 9 columns + Filters:
├─ Remove rows where MedHouseVal > 5.0
├─ Keep rows where Longitude < -119.0 (coastal regions)
└─ Drop NaN values

Shape: (10297, 9)
Data reduction: 49.9% (10,343 rows removed)

VERSION 3 (Feature-Engineered)
──────────────────────────────
Original 9 columns + 4 new features:

├─ rooms_per_household (float64)
│  └─ Formula: AveRooms × AveOccup
│  └─ Interpretation: Total rooms in typical household
│
├─ bedrooms_ratio (float64)
│  └─ Formula: AveBedrms / AveRooms
│  └─ Interpretation: Proportion of rooms that are bedrooms
│
├─ population_density (float64)
│  └─ Formula: Population / AveOccup
│  └─ Interpretation: People per household
│
└─ income_category_encoded (int64)
   └─ Formula: pd.cut(MedInc, bins=[0, 2.5, 4.5, 6.5, inf],
                       labels=['Low', 'Medium', 'High', 'Very High'])
   └─ Categories:
      ├─ 0: Low (MedInc ≤ 2.5)
      ├─ 1: Medium (2.5 < MedInc ≤ 4.5)
      ├─ 2: High (4.5 < MedInc ≤ 6.5)
      └─ 3: Very High (MedInc > 6.5)

Shape: (10297, 13)
Total features: 9 (original) + 4 (engineered)
```

---

## Key Takeaways

1. **Modularity**: Each script is independent but composable
2. **Traceability**: MLflow tracks every experiment and model
3. **Reproducibility**: DVC ensures data versioning, fixed seeds
4. **Automation**: GitHub Actions handles CI/CD pipeline
5. **Optimization**: Optuna finds best hyperparameters
6. **Comparison**: Easy comparison across models and versions
7. **Documentation**: Comprehensive guides for all operations

---

**Generated**: January 2026 | **Project**: MLOps California Housing | **Repository**: https://github.com/sloumaaaaa/mlops
