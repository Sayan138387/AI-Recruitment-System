def calculate_skill_match(required_skills, matched_skills):

    if not required_skills:
        return 0

    matched_count = len(matched_skills)

    total_required = len(required_skills)

    score = (
        matched_count /
        total_required
    ) * 100

    return score


# ---------------------------------------
# TEST
# ---------------------------------------

required_skills = [
    "python",
    "sql",
    "machine learning",
    "pandas",
    "numpy",
    "statistics"
]

matched_skills = [
    "python",
    "sql",
    "machine learning",
    "pandas",
    "numpy",
    "statistics"
]


score = calculate_skill_match(
    required_skills,
    matched_skills
)


print("Skill Match Score:")
print(f"{score:.2f}%")

# PART 2:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_job_similarity(resume_text, job_description):

    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return similarity * 100


# ---------------------------------------
# TEST
# ---------------------------------------

resume_text = """
Python developer with experience in SQL,
machine learning, Pandas, NumPy and
data analysis.
"""

job_description = """
We are looking for a Data Scientist with
experience in Python, SQL, machine learning,
Pandas, NumPy and data analysis.
"""


similarity_score = calculate_job_similarity(
    resume_text,
    job_description
)


print("\nJob Description Similarity:")
print(f"{similarity_score:.2f}%")


# PART 3:
def calculate_completeness(resume_text):

    resume_text = resume_text.lower()

    important_sections = [
        "skills",
        "education",
        "experience",
        "projects",
        "certifications",
        "contact"
    ]

    found_sections = []

    for section in important_sections:

        if section in resume_text:
            found_sections.append(section)

    score = (
        len(found_sections) /
        len(important_sections)
    ) * 100

    return score, found_sections


# ---------------------------------------
# TEST
# ---------------------------------------

resume_text = """
ALEX SHARMA

Python Developer

SKILLS
Python, SQL, Machine Learning

EDUCATION
Bachelor of Technology

PROJECTS
House Price Prediction

EXPERIENCE
Machine Learning Intern

CERTIFICATIONS
Python Programming
"""


completeness_score, found_sections = calculate_completeness(
    resume_text
)


print("\nResume Completeness:")
print(f"{completeness_score:.2f}%")

print("\nDetected Sections:")
print(", ".join(found_sections))


#  PART 4:

def calculate_ats_score(
    skill_match,
    job_similarity,
    completeness
):

    final_score = (
        (skill_match * 0.60)
        + (job_similarity * 0.25)
        + (completeness * 0.15)
    )

    return final_score

# ---------------------------------------
# FINAL ATS SCORE
# ---------------------------------------

final_score = calculate_ats_score(
    score,
    similarity_score,
    completeness_score
)


print("\n================================")
print("        FINAL ATS SCORE")
print("================================")

print(f"\nATS Score: {final_score:.2f} / 100")