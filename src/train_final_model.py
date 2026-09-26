import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

df = pd.read_csv(
    "datasets/archive (1)/Resume/resume.csv"
)


# ---------------------------------------
# 2. Clean Resume Text
# ---------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["Cleaned_Resume"] = (
    df["Resume_str"].apply(clean_text)
)


# ---------------------------------------
# 3. Separate Input and Target
# ---------------------------------------

X_text = df["Cleaned_Resume"]

y = df["Category"]


# ---------------------------------------
# 4. Split Dataset
# ---------------------------------------

X_train_text, X_test_text, y_train, y_test = train_test_split(

    X_text,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("Training Samples:", len(X_train_text))
print("Testing Samples:", len(X_test_text))


# ---------------------------------------
# 5. TF-IDF
# ---------------------------------------

vectorizer = TfidfVectorizer(

    stop_words="english",

    max_features=5000
)


X_train = vectorizer.fit_transform(
    X_train_text
)

X_test = vectorizer.transform(
    X_test_text
)


print(
    "TF-IDF Features:",
    len(vectorizer.get_feature_names_out())
)


# ---------------------------------------
# 6. Train Logistic Regression
# ---------------------------------------

model = LogisticRegression(
    max_iter=1000
)


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ---------------------------------------
# 7. Evaluate Model
# ---------------------------------------

y_pred = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nModel Accuracy:",
    accuracy
)

print(
    "Accuracy Percentage:",
    round(accuracy * 100, 2),
    "%"
)


# ---------------------------------------
# 8. Save Model
# ---------------------------------------

joblib.dump(
    model,
    "models/resume_classifier.pkl"
)


# ---------------------------------------
# 9. Save TF-IDF Vectorizer
# ---------------------------------------

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


print("\nModel saved successfully!")

print(
    "models/resume_classifier.pkl"
)

print(
    "models/tfidf_vectorizer.pkl"
)