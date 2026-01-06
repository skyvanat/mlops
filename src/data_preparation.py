import pandas as pd
import numpy as np
import yaml
import os
from pathlib import Path

def load_config():
    """Load configuration from params.yaml"""
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_raw_data(path):
    """Load raw data from CSV"""
    df = pd.read_csv(path, sep=';')
    print(f"Raw data loaded: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def clean_data(df):
    """
    Clean the dataset:
    - Remove duplicates
    - Handle missing values
    - Normalize formats
    """
    print("\n=== Data Cleaning ===")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    # Handle missing values
    print(f"Missing values before:\n{df.isnull().sum()}")
    
    # Fill numeric columns with median
    numeric_cols = ['Gold', 'Silver', 'Bronze']
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode or 'Unknown'
    categorical_cols = ['Country_Code', 'Country_Name', 'Host_city', 'Host_country']
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna('Unknown')
    
    print(f"Missing values after:\n{df.isnull().sum()}")
    
    # Create Medal column (target variable)
    if 'Medal' not in df.columns:
        # Determine medal based on Gold, Silver, Bronze
        def get_medal(row):
            if row['Gold'] > 0:
                return 'Gold'
            elif row['Silver'] > 0:
                return 'Silver'
            elif row['Bronze'] > 0:
                return 'Bronze'
            else:
                return 'None'
        
        df['Medal'] = df.apply(get_medal, axis=1)
    
    print(f"\nMedal distribution:\n{df['Medal'].value_counts()}")
    print(f"Cleaned data shape: {df.shape}")
    
    return df

def save_cleaned_data(df, path):
    """Save cleaned data"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=';', index=False)
    print(f"Cleaned data saved to {path}")

if __name__ == "__main__":
    config = load_config()
    
    # Load and clean data
    raw_path = config['paths']['raw_data']
    cleaned_path = config['paths']['cleaned_data']
    
    df = load_raw_data(raw_path)
    df_cleaned = clean_data(df)
    save_cleaned_data(df_cleaned, cleaned_path)
