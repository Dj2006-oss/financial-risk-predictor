import pandas as pd

# Load dataset
df = pd.read_csv("dataset/creditcard.csv")

# Show first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Show dataset shape
print("\nDataset shape:")
print(df.shape)

# Show column names
print("\nColumns:")
print(df.columns)

# Check class distribution
print("\nClass distribution:")
print(df['Class'].value_counts())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Basic statistics
print("\nStatistical summary:")
print(df.describe())