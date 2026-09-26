import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("datasets/archive (1)/Resume/resume.csv")


# Text cleaning function
def clean_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# Clean resumes
df["Cleaned_Resume"] = df["Resume_str"].apply(clean_text)


# Input and target
X_text = df["Cleaned_Resume"]
y = df["Category"]


# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X_text)


# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000
)


# Train the model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)


# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))