import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os

def automated_preprocessing(input_path, output_path):
    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        return
    
    df = pd.read_csv(input_path)
    print(f"Memulai preprocessing untuk: {input_path}")

    # 2. Pembersihan Data (Drop Duplicates)
    df = df.drop_duplicates()

    # 3. Handling Missing Values (SimpleImputer)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    imputer_num = SimpleImputer(strategy='mean')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # 4. Binning
    age_bins = [0, 30, 50, 100]
    age_labels = ['Young', 'Middle', 'Senior']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    income_bins = [0, 40000, 80000, df['income'].max() + 1] 
    income_labels = ['Low', 'Medium', 'High']
    df['income_group'] = pd.cut(df['income'], bins=income_bins, labels=income_labels)

    # 5. Encoding
    le = LabelEncoder()
    cat_cols = ['gender', 'occupation', 'marital_status', 'age_group', 'income_group', 'loan_status']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    edu_mapping = {'High School': 1, "Associate's": 2, "Bachelor's": 3, "Master's": 4, 'Doctoral': 5}
    df['education_level'] = df['education_level'].map(edu_mapping).fillna(0)

    # 6. Standardization (Scaling)
    scaler = StandardScaler()
    target_col = 'loan_status'
    features = df.drop(columns=[target_col])
    
    scaled_features = scaler.fit_transform(features)
    df_final = pd.DataFrame(scaled_features, columns=features.columns)
    
    df_final[target_col] = df[target_col].values

    # 7. Simpan Hasil
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df_final.to_csv(output_path, index=False)
    print(f"Berhasil! Data siap latih disimpan di: {output_path}")

if __name__ == "__main__":
    automated_preprocessing(
        input_path='loan_data_raw.csv', 
        output_path='preprocessing/loan_data_preprocessed.csv'
    )
