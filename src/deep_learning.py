import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------
# 1. Load Dataset
# ---------------------------------------

df = pd.read_csv(
    "datasets/archive (1)/Resume/resume.csv"
)


# ---------------------------------------
# 2. Text Cleaning
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


df["Cleaned_Resume"] = df["Resume_str"].apply(
    clean_text
)


# ---------------------------------------
# 3. Separate Input and Target
# ---------------------------------------

X_text = df["Cleaned_Resume"]

y = df["Category"]


# ---------------------------------------
# 4. TF-IDF
# ---------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000
)

X = vectorizer.fit_transform(X_text)


print("TF-IDF Matrix Shape:")
print(X.shape)


# ---------------------------------------
# 5. Train-Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


print("\nTraining Samples:")
print(X_train.shape[0])

print("\nTesting Samples:")
print(X_test.shape[0])


# ---------------------------------------
# 6. Create Neural Network
# ---------------------------------------

model = MLPClassifier(

    hidden_layer_sizes=(128, 64),

    activation="relu",

    solver="adam",

    max_iter=100,

    random_state=42,

    early_stopping=True
)


# ---------------------------------------
# 7. Train Model
# ---------------------------------------

print("\nTraining Neural Network...")

model.fit(
    X_train,
    y_train
)

print("Neural Network Training Completed!")


# ---------------------------------------
# 8. Predictions
# ---------------------------------------

y_pred = model.predict(
    X_test
)


# ---------------------------------------
# 9. Evaluation
# ---------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nNeural Network Accuracy:")
print(accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)