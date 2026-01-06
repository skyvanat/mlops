import pandas as pd
import numpy as np
import yaml
import os

def load_config():
    """Load configuration from params.yaml"""
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_cleaned_data(path):
    """Load cleaned data"""
    df = pd.read_csv(path, sep=';')
    print(f"Cleaned data loaded: {df.shape}")
    return df

def engineer_features(df):
    """
    Advanced feature engineering:
    - Add historical statistics per country
    - Add participation count
    - Normalize medal counts
    - Add host advantage
    """
    print("\n=== Feature Engineering ===")
    df = df.copy()
    
    # 1. Total medals per country (historical)
    country_stats = df.groupby('Country_Name')[['Gold', 'Silver', 'Bronze']].sum()
    country_stats['Total_Medals'] = country_stats['Gold'] + country_stats['Silver'] + country_stats['Bronze']
    country_stats['Medal_Ratio'] = country_stats['Gold'] / (country_stats['Total_Medals'] + 1)  # Avoid division by 0
    
    # Merge with main df
    df = df.merge(country_stats.add_prefix('Historical_'), left_on='Country_Name', right_index=True, how='left')
    
    # 2. Performance per year (cumulative)
    year_stats = df.groupby('Year')[['Gold', 'Silver', 'Bronze']].sum()
    year_stats['Year_Total'] = year_stats.sum(axis=1)
    df = df.merge(year_stats.add_prefix('Year_'), left_on='Year', right_index=True, how='left')
    
    # 3. Host advantage: Did the country host the Olympics?
    df['Is_Host'] = (df['Country_Name'] == df['Host_country']).astype(int)
    
    # 4. Medal count normalization
    df['Total_Medals'] = df['Gold'] + df['Silver'] + df['Bronze']
    df['Medal_Strength'] = df['Total_Medals'] / (df['Historical_Total_Medals'] + 1)
    
    # 5. Encode Medal (target)
    medal_mapping = {'Gold': 1, 'Silver': 2, 'Bronze': 3, 'None': 0}
    df['Medal_Encoded'] = df['Medal'].map(medal_mapping)
    
    # 6. Encode categorical features
    df['Country_Code_Encoded'] = pd.Categorical(df['Country_Code']).codes
    df['Host_Country_Encoded'] = pd.Categorical(df['Host_country']).codes
    
    # Fill any NaN values
    df = df.fillna(0)
    
    print(f"\nNew features created:")
    print(f"- Historical statistics (Gold, Silver, Bronze, Total, Ratio)")
    print(f"- Year statistics (Total medals per year)")
    print(f"- Is_Host (binary)")
    print(f"- Medal_Strength (normalized)")
    print(f"- Medal_Encoded (target encoded)")
    print(f"- Country_Code_Encoded, Host_Country_Encoded")
    
    print(f"\nFeatured data shape: {df.shape}")
    print(f"New columns: {len(df.columns)}")
    
    return df

def save_featured_data(df, path):
    """Save featured data"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=';', index=False)
    print(f"Featured data saved to {path}")

if __name__ == "__main__":
    config = load_config()
    
    cleaned_path = config['paths']['cleaned_data']
    featured_path = config['paths']['featured_data']
    
    df = load_cleaned_data(cleaned_path)
    df_featured = engineer_features(df)
    save_featured_data(df_featured, featured_path)
