import pandas as pd


# Load Job Description dataset
df = pd.read_csv(
    "datasets/Job_Description/job_title_des.csv"
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