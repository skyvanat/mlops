import pandas as pd
import numpy as np
import yaml
import os
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import mlflow.sklearn

def load_config():
    """Load configuration from params.yaml"""
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_featured_data(path):
    """Load featured data"""
    df = pd.read_csv(path, sep=';')
    print(f"Featured data loaded: {df.shape}")
    return df

def check_class_balance(y):
    """
    Analyze class balance and provide information
    """
    print("\n=== Class Balance Analysis ===")
    distribution = pd.Series(y).value_counts()
    print("Class distribution:")
    print(distribution)
    print("\nPercentage distribution:")
    print(distribution / len(y) * 100)
    
    # Calculate balance ratio
    if len(distribution) > 1:
        imbalance_ratio = distribution.max() / distribution.min()
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}x")
        
        if imbalance_ratio > 2:
            print("SIGNIFICANT CLASS IMBALANCE DETECTED!")
            print("Using class_weight='balanced' to adjust")
            return True
    return False

def prepare_features(df):
    """
    Prepare features and target for training
    Remove non-predictive columns and avoid data leakage
    """
    print("\n=== Feature Preparation ===")
    print("WARNING: Using minimal features (avoiding data leakage)")
    
    # Select relevant features for training - Only historical performance
    # This avoids all data leakage from current medals
    feature_cols = [
        'Year', 'Is_Host', 
        'Historical_Total_Medals',  # Only total historical medals
        'Country_Code_Encoded', 'Host_Country_Encoded'
    ]
    
    # Filter only existing columns
    available_cols = [col for col in feature_cols if col in df.columns]
    print(f"Using {len(available_cols)} features for training")
    print(f"Features: {available_cols}")
    
    X = df[available_cols].copy()
    y = df['Medal_Encoded'].copy()
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, available_cols

def train_model(X_train, y_train, config):
    """
    Train Random Forest model with balanced class weights
    """
    print("\n=== Model Training ===")
    
    # Check class balance
    has_imbalance = check_class_balance(y_train)
    
    # Calculate class weights to handle imbalance
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"\nClass weights: {class_weight_dict}")
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=config['model']['n_estimators'],
        max_depth=config['model']['max_depth'],
        min_samples_split=config['model'].get('min_samples_split', 5),
        min_samples_leaf=config['model'].get('min_samples_leaf', 2),
        random_state=config['model']['random_state'],
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print(f"\nModel trained successfully")
    
    return model, class_weight_dict

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """
    Evaluate model on train and test sets
    """
    print("\n=== Model Evaluation ===")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print(f"\nTrain Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Train F1-Score: {train_f1:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    print("\n=== Classification Report (Test Set) ===")
    unique_labels = np.unique(y_test)
    label_names = {0: 'None', 1: 'Gold', 2: 'Silver', 3: 'Bronze'}
    target_names = [label_names.get(i, f'Class_{i}') for i in unique_labels]
    print(classification_report(y_test, y_test_pred, 
                               labels=unique_labels,
                               target_names=target_names,
                               zero_division=0))
    
    # Feature importance
    print("\n=== Top 10 Important Features ===")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    metrics = {
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'train_f1': float(train_f1),
        'test_f1': float(test_f1)
    }
    
    return metrics, y_test_pred

def save_model(model, path):
    """Save trained model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def save_metrics(metrics, path):
    """Save metrics to JSON"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {path}")

if __name__ == "__main__":
    # Load configuration
    config = load_config()
    
    # Set MLflow experiment
    mlflow.set_experiment("Olympic_Medals_Prediction")
    
    with mlflow.start_run():
        print("=" * 50)
        print("OLYMPIC MEDALS PREDICTION - TRAINING PIPELINE")
        print("=" * 50)
        
        # Load data
        featured_path = config['paths']['featured_data']
        df = load_featured_data(featured_path)
        
        # Prepare features
        X, y, feature_names = prepare_features(df)
        
        # Split data
        print("\n=== Data Splitting ===")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config['train']['test_size'],
            random_state=config['train']['random_state'],
            stratify=y
        )
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Log hyperparameters
        mlflow.log_params({
            'n_estimators': config['model']['n_estimators'],
            'max_depth': config['model']['max_depth'],
            'min_samples_split': config['model'].get('min_samples_split', 5),
            'min_samples_leaf': config['model'].get('min_samples_leaf', 2),
            'test_size': config['train']['test_size'],
            'random_state': config['model']['random_state'],
            'class_weight': 'balanced'
        })
        
        # Train model with automatic class weight balancing
        model, class_weights = train_model(X_train, y_train, config)
        
        # Evaluate model
        metrics, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Save model and metrics
        model_path = config['paths']['model']
        metrics_path = config['paths']['metrics']
        
        save_model(model, model_path)
        save_metrics(metrics, metrics_path)
        
        # Log artifacts to MLflow (skip if permission denied in CI/CD)
        try:
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metrics_path)
        except PermissionError:
            print("WARNING: Could not log artifacts to MLflow (permission denied)")
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
