
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------
# PART 1: SKILL MATCHING
# ---------------------------------------

def calculate_skill_match(required_skills, matched_skills):

    required = {
        skill.strip().lower()
        for skill in required_skills
        if skill.strip()
    }

    matched = {
        skill.strip().lower()
        for skill in matched_skills
        if skill.strip()
    }

    if not required:
        return 0.0

    matched_count = len(required.intersection(matched))

    return (matched_count / len(required)) * 100


# ---------------------------------------
# PART 2: JOB DESCRIPTION SIMILARITY
# ---------------------------------------

def calculate_job_similarity(resume_text, job_description):

    if not resume_text.strip() or not job_description.strip():
        return 0.0

    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # Handles text containing no usable words.
        return 0.0

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    return float(similarity * 100)


# ---------------------------------------
# PART 3: RESUME COMPLETENESS
# ---------------------------------------

def calculate_completeness(resume_text):

    text = resume_text.lower()

    section_patterns = {
        "Skills": r"\bskills?\b",
        "Education": (
            r"\b(education|bachelor|master|degree|"
            r"university|college)\b"
        ),
        "Experience": (
            r"\b(experience|employment|internship|"
            r"intern|worked)\b"
        ),
        "Projects": r"\bprojects?\b",
        "Certifications": r"\b(certifications?|certificates?)\b",
        "Contact": (
            r"\bcontact\b|"
            r"\b[\w.+%-]+@[\w.-]+\.[a-zA-Z]{2,}\b|"
            r"\blinkedin\.com\b|"
            r"\bgithub\.com\b"
        )
    }

    found_sections = []

    for section, pattern in section_patterns.items():

        if re.search(pattern, text):
            found_sections.append(section)

    total_sections = len(section_patterns)

    score = (
        len(found_sections) / total_sections
    ) * 100

    return score, found_sections


# ---------------------------------------
# PART 4: FINAL ATS SCORE
# ---------------------------------------

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
