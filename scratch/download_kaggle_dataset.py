import os
import shutil

import kagglehub

print("Downloading real Kaggle Crop Yield dataset via kagglehub...")
path = kagglehub.dataset_download("patelrishabh/crop-yield-prediction-dataset")
print(f"Downloaded to: {path}")

files = os.listdir(path)
print(f"Files in dataset path: {files}")

csv_file = None
for f in files:
    if f.endswith(".csv"):
        csv_file = os.path.join(path, f)
        break

if csv_file and os.path.exists(csv_file):
    target_path = "data/raw/crop_yield_raw.csv"
    os.makedirs("data/raw", exist_ok=True)
    shutil.copy(csv_file, target_path)
    print(f"Successfully copied real dataset {csv_file} to {target_path}!")

    import pandas as pd

    df = pd.read_csv(target_path)
    print(f"Real Dataset Shape: {df.shape}")
    print(f"Real Dataset Columns: {list(df.columns)}")
else:
    print("No CSV file found in downloaded dataset directory!")
