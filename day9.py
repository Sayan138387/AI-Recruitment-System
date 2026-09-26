import pandas as pd
import re

# Load the real Resume Dataset
df = pd.read_csv("datasets/archive (1)/Resume/resume.csv")


# Function to clean resume text
def clean_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply text cleaning
df["Cleaned_Resume"] = df["Resume_str"].apply(clean_text)


# Display original and cleaned resume
print("Original Resume:")
print(df["Resume_str"].iloc[0])

print("\n\nCleaned Resume:")
print(df["Cleaned_Resume"].iloc[0])