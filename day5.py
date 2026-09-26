import pandas as pd

# Read the CSV file
df = pd.read_csv("datasets/sample_resumes.csv")

# Display the entire dataset
print(df)

# Show the first five rows
print("\nFirst Five Rows:")
print(df.head())

# Show the last five rows
print("\nLast Five Rows:")
print(df.tail())

# Display the shape
print("\nShape:")
print(df.shape)

# Display the column names
print("\nColumns:")
print(df.columns)

# Display dataset information
print("\nInfo:")
df.info()

# Display statistics
print("\nStatistics:")
print(df.describe())

# Display category counts
print("\nCategory Count:")
print(df["Category"].value_counts())


##### DAY 6 CODE   #####*************************************

import pandas as pd

# Read dataset
df = pd.read_csv("datasets/sample_resumes.csv")

# Select one column
print(df["Skills"])

# Select multiple columns
print(df[["Name", "Skills"]])

# First row
print(df.iloc[0])

# First three rows
print(df.iloc[0:3])

# Filter Experience > 2
print(df[df["Experience"] > 2])

# Filter Data Science candidates
print(df[df["Category"] == "Data Science"])

# Multiple conditions
print(df[(df["Experience"] >= 2) & (df["Category"] == "Data Science")])

# Sort by experience
print(df.sort_values(by="Experience", ascending=False))