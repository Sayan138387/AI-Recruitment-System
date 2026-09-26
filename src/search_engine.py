import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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


# Create TF-IDF representation
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

resume_vectors = vectorizer.fit_transform(df["Cleaned_Resume"])


def search_resumes(query, top_n=5):

    # Convert search query into TF-IDF
    query_vector = vectorizer.transform([clean_text(query)])

    # Calculate similarity
    similarity_scores = cosine_similarity(
        query_vector,
        resume_vectors
    ).flatten()

    # Get top results
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    results = df.iloc[top_indices].copy()

    results["Similarity_Score"] = similarity_scores[top_indices]

    return results[
        ["ID", "Category", "Similarity_Score"]
    ]


# Test search
if __name__ == "__main__":

    query = "python machine learning sql"

    results = search_resumes(query)

    print("\nSearch Query:")
    print(query)

    print("\nTop Matching Resumes:")
    print(results)