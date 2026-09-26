import pandas as pd

# Load dataset
df = pd.read_csv("datasets/sample_resumes.csv")

print("Original Dataset:")
print(df)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Convert Skills to lowercase
df["Skills"] = df["Skills"].str.lower()

# Remove extra spaces
df["Skills"] = df["Skills"].str.strip()

print("\nCleaned Dataset:")
print(df)