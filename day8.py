import pandas as pd

# Load the real Resume Dataset
df = pd.read_csv("datasets/archive (1)/Resume/resume.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display information about the dataset
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Count resumes in each category
print("\nResume Count by Category:")
print(df["Category"].value_counts())