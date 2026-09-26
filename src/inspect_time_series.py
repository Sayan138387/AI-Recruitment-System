import pandas as pd

df = pd.read_csv(
    "datasets/time_series/jobpost/data_job_posts.csv"
)

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())