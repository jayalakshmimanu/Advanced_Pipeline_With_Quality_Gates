from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

# Load dataset
data = load_breast_cancer(as_frame=True)

# Features
df = data.frame

# Rename target column for clarity
df.rename(columns={"target": "diagnosis"}, inplace=True)

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save CSV
df.to_csv("data/breast_cancer.csv", index=False)

print("Dataset saved successfully!")
print(df.head())