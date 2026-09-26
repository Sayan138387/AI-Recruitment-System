import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer


# Load the Resume Dataset
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


# Clean all resumes
df["Cleaned_Resume"] = df["Resume_str"].apply(clean_text)


# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)


# Convert resume text into numerical features
X = vectorizer.fit_transform(df["Cleaned_Resume"])


# Target variable
y = df["Category"]


# Display results
print("Original Number of Resumes:")
print(len(df))

print("\nNumber of TF-IDF Features:")
print(len(vectorizer.get_feature_names_out()))

print("\nTF-IDF Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFirst 20 TF-IDF Features:")
print(vectorizer.get_feature_names_out()[:20])