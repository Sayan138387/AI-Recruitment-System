import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(
    "datasets/time_series/jobpost/data_job_posts.csv"
)

# Create a proper monthly date using Year and Month
df["Date"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month"],
        day=1
    )
)

# Count job postings per month
monthly_jobs = (
    df.groupby("Date")
    .size()
)

print("\nMonthly Job Postings:")
print(monthly_jobs)

print("\nTotal Months:")
print(len(monthly_jobs))

# Plot time-series
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_jobs.index,
    monthly_jobs.values
)

plt.xlabel("Date")
plt.ylabel("Number of Job Postings")

plt.title(
    "Job Posting Trend Over Time"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ---------------------------------------
# IT Job Demand Over Time
# ---------------------------------------

it_jobs = df[df["IT"] == True]

it_monthly_jobs = (
    it_jobs.groupby("Date")
    .size()
)

print("\nMonthly IT Job Postings:")
print(it_monthly_jobs)

print("\nTotal IT Job Postings:")
print(len(it_jobs))

# Plot IT job demand
plt.figure(figsize=(12, 6))

plt.plot(
    it_monthly_jobs.index,
    it_monthly_jobs.values
)

plt.xlabel("Date")
plt.ylabel("Number of IT Job Postings")

plt.title(
    "IT Job Demand Trend Over Time"
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()