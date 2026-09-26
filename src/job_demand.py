import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------
# 1. Load Job Description Dataset
# ---------------------------------------

df = pd.read_csv(
    "datasets/Job_Description/job_title_des.csv"
)


# ---------------------------------------
# 2. Remove unnecessary index column
# ---------------------------------------

df = df.drop(
    columns=["Unnamed: 0"]
)


# ---------------------------------------
# 3. Count Job Titles
# ---------------------------------------

job_counts = (
    df["Job Title"]
    .value_counts()
)


print("\nTop 20 Job Titles:")

print(
    job_counts.head(20)
)


# ---------------------------------------
# 4. Plot Top 15 Job Titles
# ---------------------------------------

top_jobs = job_counts.head(15)


plt.figure(figsize=(12, 6))

top_jobs.sort_values().plot(
    kind="barh"
)

plt.xlabel("Number of Job Postings")
plt.ylabel("Job Title")

plt.title(
    "Most Frequent Job Roles in Dataset"
)

plt.tight_layout()

plt.show()