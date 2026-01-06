import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path, sep=';')

def basic_cleaning(df):
    df = df.drop_duplicates()
    # Fill numeric columns with median
    numeric_cols = ['Gold', 'Silver', 'Bronze']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    # Fill categorical columns
    df['Country_Code'] = df['Country_Code'].fillna('Unknown')
    df['Country_Name'] = df['Country_Name'].fillna('Unknown')
    return df

def encode_features(df):
    le_country = LabelEncoder()
    df['Country_Code_Encoded'] = le_country.fit_transform(df['Country_Code'])
    df['Host_country_Encoded'] = df['Host_country'].astype('category').cat.codes
    return df

# Load and process the dataset
if __name__ == "__main__":
    file_path = 'Country_Medals.csv'
    
    # Load data
    df = load_data(file_path)
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData types:\n{df.dtypes}")
    
    # Clean data
    df_cleaned = basic_cleaning(df)
    print(f"\nCleaned data shape: {df_cleaned.shape}")
    
    # Encode features
    df_processed = encode_features(df_cleaned)
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"\nProcessed data sample:\n{df_processed.head()}")
    print("\nPreprocessing completed successfully!")
