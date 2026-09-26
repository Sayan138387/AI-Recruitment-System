import re
import joblib

from pdf_parser import extract_text_from_pdf
from role_matcher import find_best_role


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


# --------------------------------
# LOAD MODEL
# --------------------------------

model = joblib.load(
    "models/resume_classifier.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# --------------------------------
# EXTRACT RESUME
# --------------------------------

pdf_path = "resumes/test_resume.pdf"

resume_text = extract_text_from_pdf(
    pdf_path
)


# --------------------------------
# CLEAN RESUME
# --------------------------------

cleaned_resume = clean_text(
    resume_text
)


# --------------------------------
# ML PREDICTION
# --------------------------------

resume_vector = vectorizer.transform(
    [cleaned_resume]
)

prediction = model.predict(
    resume_vector
)

predicted_category = prediction[0]


# --------------------------------
# TOP 3 PREDICTIONS
# --------------------------------

probabilities = model.predict_proba(
    resume_vector
)[0]

classes = model.classes_

top_indices = probabilities.argsort()[::-1][:3]


# --------------------------------
# ROLE MATCHING
# --------------------------------

role_results = find_best_role(
    cleaned_resume
)


# --------------------------------
# DISPLAY RESULTS
# --------------------------------

print("\n================================")
print("      RESUME ANALYSIS")
print("================================")


print("\nML Predicted Category:")
print(predicted_category)


print("\nTop 3 ML Predictions:")

for index in top_indices:

    print(
        f"{classes[index]}: "
        f"{probabilities[index] * 100:.2f}%"
    )


print("\n--------------------------------")
print("      ROLE MATCHING")
print("--------------------------------")


print("\nTop 3 Suitable Roles:")

for result in role_results[:3]:

    print(
        f"\nRole: {result['role']}"
    )

    print(
        f"Skill Match: "
        f"{result['score']:.2f}%"
    )

    if result["matched_skills"]:

        print(
            "Matched Skills: "
            + ", ".join(
                result["matched_skills"]
            )
        )

    else:

        print(
            "Matched Skills: None"
        )


print("\n================================")