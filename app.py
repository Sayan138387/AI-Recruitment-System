
import io
import json
import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.ats_score import (
    calculate_skill_match,
    calculate_job_similarity,
    calculate_completeness,
    calculate_ats_score,
)
from src.pdf_parser import extract_text_from_pdf


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "resume_classifier.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
KNOWLEDGE_PATH = BASE_DIR / "datasets" / "job_skill_knowledge.json"

st.set_page_config(
    page_title="AI Recruitment System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1350px;
        }

        [data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.20);
            padding: 18px;
            border-radius: 12px;
        }

        .hero {
            padding: 28px;
            border-radius: 16px;
            background: linear-gradient(120deg, #15314b, #14665d);
            color: white;
            margin-bottom: 24px;
        }

        .hero h1 {
            color: white;
            margin: 0 0 10px 0;
            font-size: 2rem;
        }

        .hero p {
            color: #e0f2f1;
            margin: 0;
            font-size: 1rem;
        }

        div[data-testid="stProgress"] {
            margin-bottom: 8px;
        }

        .small-note {
            font-size: 0.9rem;
            opacity: 0.8;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. LOAD KNOWLEDGE BASE AND TRAINED MODELS
# ============================================================

@st.cache_data
def load_knowledge_base():

    with open(
        KNOWLEDGE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@st.cache_resource
def load_models():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


knowledge_base = load_knowledge_base()


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text,
    )

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================
# 5. SKILL MATCHING
# ============================================================

def get_known_skills():

    skills = {
        skill.strip()
        for role_data in knowledge_base.values()
        for skill in role_data["required_skills"]
        if skill.strip()
    }

    return sorted(
        skills,
        key=lambda skill: (-len(skill), skill.lower()),
    )


def find_matched_skills(text, required_skills):

    normalized_text = re.sub(
        r"\s+",
        " ",
        text.lower(),
    )

    matched_skills = []

    for skill in required_skills:

        normalized_skill = re.sub(
            r"\s+",
            " ",
            skill.strip().lower(),
        )

        if not normalized_skill:
            continue

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(normalized_skill)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, normalized_text):
            matched_skills.append(skill)

    return matched_skills


# ============================================================
# 6. ROLE RANKING
# ============================================================

def rank_resume_roles(resume_text):

    results = []

    for role, data in knowledge_base.items():

        required = data["required_skills"]

        matched = find_matched_skills(
            resume_text,
            required,
        )

        score = calculate_skill_match(
            required,
            matched,
        )

        results.append(
            {
                "role": role,
                "score": score,
                "matched_skills": matched,
                "missing_skills": [
                    skill
                    for skill in required
                    if skill not in matched
                ],
            }
        )

    results.sort(
        key=lambda result: (
            -result["score"],
            result["role"],
        )
    )

    return results


# ============================================================
# 7. RESUME ANALYSIS ENGINE
# ============================================================

def analyze_resume(uploaded_file, selected_role, job_description):

    # Extract text from the uploaded PDF.
    pdf_bytes = io.BytesIO(
        uploaded_file.getvalue()
    )

    resume_text = extract_text_from_pdf(
        pdf_bytes
    )

    if not resume_text.strip():

        raise ValueError(
            "No readable text was extracted. "
            "The PDF may be scanned or image-based."
        )

    cleaned_resume = clean_text(resume_text)

    if not cleaned_resume:

        raise ValueError(
            "The resume does not contain usable text."
        )

    # Run the existing trained ML classifier.
    model, vectorizer = load_models()

    resume_vector = vectorizer.transform(
        [cleaned_resume]
    )

    predicted_category = model.predict(
        resume_vector
    )[0]

    probabilities = model.predict_proba(
        resume_vector
    )[0]

    classes = model.classes_

    top_indices = probabilities.argsort()[::-1][:3]

    top_predictions = [
        {
            "category": str(classes[index]),
            "score": float(probabilities[index] * 100),
        }
        for index in top_indices
    ]

    # Detect recognizable skills in the job description.
    jd_detected_skills = find_matched_skills(
        job_description,
        get_known_skills(),
    )

    # Use job-description skills when available.
    # Otherwise, fall back to the selected role's defaults.
    if jd_detected_skills:

        required_skills = jd_detected_skills

        requirements_source = (
            "Detected from the job description"
        )

    else:

        required_skills = knowledge_base[
            selected_role
        ]["required_skills"]

        requirements_source = (
            "Default requirements for " + selected_role
        )

    # Compare the resume with those requirements.
    matched_skills = find_matched_skills(
        resume_text,
        required_skills,
    )

    missing_skills = [
        skill
        for skill in required_skills
        if skill not in matched_skills
    ]

    # Calculate ATS components.
    skill_score = calculate_skill_match(
        required_skills,
        matched_skills,
    )

    similarity_score = calculate_job_similarity(
        resume_text,
        job_description,
    )

    completeness_score, found_sections = (
        calculate_completeness(resume_text)
    )

    final_score = calculate_ats_score(
        skill_score,
        similarity_score,
        completeness_score,
    )

    # Rank roles using the knowledge base.
    role_results = rank_resume_roles(
        resume_text
    )

    # Keep every result together for display and reporting.
    return {
        "filename": Path(uploaded_file.name).name,
        "selected_role": selected_role,
        "analyzed_at": datetime.now().strftime(
            "%d %b %Y, %I:%M %p"
        ),
        "resume_text": resume_text,
        "predicted_category": str(predicted_category),
        "top_predictions": top_predictions,
        "required_skills": required_skills,
        "requirements_source": requirements_source,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_score": skill_score,
        "similarity_score": similarity_score,
        "completeness_score": completeness_score,
        "final_score": final_score,
        "found_sections": found_sections,
        "role_results": role_results,
    }


# ============================================================
# 8. DOWNLOADABLE REPORT
# ============================================================

def create_report(result):

    lines = [
        "AI RECRUITMENT AND RESUME SCREENING SYSTEM",
        "=" * 48,
        "",
        f"Resume file: {result['filename']}",
        f"Target role: {result['selected_role']}",
        f"Analysis date: {result['analyzed_at']}",
        "",
        "ATS SCORE SUMMARY",
        "-" * 30,
        f"Final ATS score: {result['final_score']:.2f}/100",
        f"Skill match: {result['skill_score']:.2f}%",
        (
            "Job description similarity: "
            f"{result['similarity_score']:.2f}%"
        ),
        (
            "Resume completeness: "
            f"{result['completeness_score']:.2f}%"
        ),
        "",
        "SCORING FORMULA",
        "-" * 30,
        "60% skill match + 25% job similarity "
        "+ 15% resume completeness",
        "",
        "ML CLASSIFICATION",
        "-" * 30,
        f"Predicted category: {result['predicted_category']}",
        "",
        "Top category outputs:",
    ]

    for prediction in result["top_predictions"]:

        lines.append(
            f"- {prediction['category']}: "
            f"{prediction['score']:.2f}%"
        )

    lines.extend(
        [
            "",
            "SKILL REQUIREMENTS",
            "-" * 30,
            f"Requirements source: {result['requirements_source']}",
            "Required skills: "
            + (
                ", ".join(result["required_skills"])
                or "None detected"
            ),
            "Matched skills: "
            + (
                ", ".join(result["matched_skills"])
                or "None detected"
            ),
            "Skills not detected: "
            + (
                ", ".join(result["missing_skills"])
                or "None"
            ),
            "",
            "RESUME COMPLETENESS",
            "-" * 30,
            "Detected indicators: "
            + (
                ", ".join(result["found_sections"])
                or "None"
            ),
            "",
            "OTHER SUITABLE ROLES",
            "-" * 30,
        ]
    )

    for role in result["role_results"][:3]:

        lines.append(
            f"- {role['role']}: {role['score']:.2f}% skill match"
        )

        lines.append(
            "  Matched skills: "
            + (
                ", ".join(role["matched_skills"])
                or "None"
            )
        )

    lines.extend(
        [
            "",
            "IMPORTANT LIMITATIONS",
            "-" * 30,
            (
                "The ATS score is a project-specific heuristic, "
                "not an official industry-standard score."
            ),
            (
                "Skill detection relies on known skill terms. "
                "Text similarity and completeness checks are "
                "approximate measures."
            ),
            (
                "ML category scores describe the model output, "
                "not the probability of getting hired."
            ),
            (
                "This report is intended for decision support, "
                "not as a standalone hiring decision."
            ),
        ]
    )

    return "\n".join(lines)


# ============================================================
# 9. PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>AI Recruitment &amp; Resume Screening</h1>
        <p>
            Analyze resumes, identify skill gaps, compare job
            requirements, and explore suitable roles.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 10. SIDEBAR INPUT PANEL
# ============================================================

st.sidebar.title("Resume Analyzer")

st.sidebar.caption(
    "Provide a resume and the target job requirements "
    "to begin the analysis."
)

with st.sidebar.form("resume_analysis_form"):

    uploaded_file = st.file_uploader(
        "Upload resume (PDF)",
        type=["pdf"],
        help="Text-based PDFs are best supported.",
    )

    available_roles = sorted(
        knowledge_base.keys()
    )

    selected_role = st.selectbox(
        "Target job role",
        available_roles,
    )

    st.caption(
        "Default skills: "
        + ", ".join(
            knowledge_base[selected_role]["required_skills"]
        )
    )

    job_description = st.text_area(
        "Job description",
        height=220,
        placeholder=(
            "Paste the job responsibilities, skills, "
            "qualifications, and requirements..."
        ),
    )

    submit_analysis = st.form_submit_button(
        "Analyze Resume",
        type="primary",
        use_container_width=True,
    )


st.sidebar.divider()

st.sidebar.subheader("Scoring Method")

st.sidebar.write("**60%** — Skill match")
st.sidebar.write("**25%** — Job-description similarity")
st.sidebar.write("**15%** — Resume completeness")

st.sidebar.caption(
    "This weighting is defined for this academic project "
    "and is not an official ATS industry standard."
)


# ============================================================
# 11. SESSION STATE AND FORM SUBMISSION
# ============================================================

if "analysis_results" not in st.session_state:

    st.session_state.analysis_results = None

if "analysis_error" not in st.session_state:

    st.session_state.analysis_error = None


if submit_analysis:

    # Clear stale results before processing new inputs.
    st.session_state.analysis_results = None
    st.session_state.analysis_error = None

    if uploaded_file is None:

        st.session_state.analysis_error = (
            "Please upload a PDF resume."
        )

    elif not job_description.strip():

        st.session_state.analysis_error = (
            "Please enter the job description."
        )

    else:

        try:

            with st.spinner(
                "Extracting text and analyzing the resume..."
            ):

                result = analyze_resume(
                    uploaded_file,
                    selected_role,
                    job_description,
                )

            st.session_state.analysis_results = result

        except FileNotFoundError as error:

            st.session_state.analysis_error = (
                "A required project file is missing. "
                "Check the models and knowledge-base dataset."
                f"\n\nDetails: {error}"
            )

        except Exception as error:

            st.session_state.analysis_error = (
                f"{type(error).__name__}: {error}"
            )


# ============================================================
# 12. DISPLAY ERRORS
# ============================================================

if st.session_state.analysis_error:

    st.error(
        st.session_state.analysis_error
    )


# ============================================================
# 13. EMPTY STATE
# ============================================================

result = st.session_state.analysis_results

if result is None and not st.session_state.analysis_error:

    st.subheader("Welcome to your resume screening dashboard")

    st.write(
        "Upload a resume, choose a target role, and provide "
        "a job description in the sidebar to begin."
    )

    feature_col1, feature_col2, feature_col3 = st.columns(3)

    with feature_col1:

        st.markdown("### 🧠 ML Classification")

        st.write(
            "Predict a broad resume category using the "
            "trained machine learning model."
        )

    with feature_col2:

        st.markdown("### 🎯 Skill Matching")

        st.write(
            "Compare candidate skills against recognizable "
            "requirements and identify skills not detected."
        )

    with feature_col3:

        st.markdown("### 📊 ATS Analysis")

        st.write(
            "Review the weighted score, job similarity, "
            "resume completeness, and role matches."
        )

    st.info(
        "The analysis provides decision-support metrics. "
        "It does not make a final hiring decision."
    )


# ============================================================
# 14. RESULTS DASHBOARD
# ============================================================

if result is not None:

    st.caption(
        f"Analyzed file: {result['filename']}  |  "
        f"Target role: {result['selected_role']}  |  "
        f"Analyzed at: {result['analyzed_at']}"
    )

    st.header("Resume Analysis Dashboard")

    # ----------------------------------------
    # Key metrics
    # ----------------------------------------

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Final ATS Score",
        f"{result['final_score']:.2f}/100",
    )

    metric2.metric(
        "Skill Match",
        f"{result['skill_score']:.2f}%",
    )

    metric3.metric(
        "Job Similarity",
        f"{result['similarity_score']:.2f}%",
    )

    metric4.metric(
        "Completeness",
        f"{result['completeness_score']:.2f}%",
    )

    st.progress(
        max(
            0.0,
            min(
                1.0,
                result["final_score"] / 100,
            ),
        )
    )

    st.caption(
        "The final score is a weighted project metric. "
        "It is not a hiring probability."
    )

    st.divider()

    # ----------------------------------------
    # Download report
    # ----------------------------------------

    report_text = create_report(result)

    report_filename = (
        "resume_analysis_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".txt"
    )

    st.download_button(
        label="⬇️ Download Analysis Report",
        data=report_text,
        file_name=report_filename,
        mime="text/plain",
        use_container_width=False,
    )

    # ----------------------------------------
    # Tabs
    # ----------------------------------------

    overview_tab, skills_tab, roles_tab, text_tab = st.tabs(
        [
            "📊 Overview",
            "🎯 Skills & Completeness",
            "💼 Role Matches",
            "📄 Resume Text",
        ]
    )

    # ========================================
    # OVERVIEW TAB
    # ========================================

    with overview_tab:

        st.subheader("Machine Learning Classification")

        st.write(
            f"**Predicted category:** "
            f"{result['predicted_category']}"
        )

        prediction_df = pd.DataFrame(
            [
                {
                    "Category": prediction["category"],
                    "Model Score": (
                        f"{prediction['score']:.2f}%"
                    ),
                }
                for prediction in result["top_predictions"]
            ]
        )

        st.dataframe(
            prediction_df,
            hide_index=True,
            use_container_width=True,
        )

        if result["top_predictions"][0]["score"] < 25:

            st.warning(
                "The top category has a relatively low model "
                "score. Review the resume manually instead "
                "of relying on this classification alone."
            )

        st.subheader("Score Breakdown")

        score_df = pd.DataFrame(
            {
                "Score (%)": [
                    result["skill_score"],
                    result["similarity_score"],
                    result["completeness_score"],
                ]
            },
            index=[
                "Skill Match",
                "Job Similarity",
                "Resume Completeness",
            ],
        )

        st.bar_chart(
            score_df,
            height=280,
        )

        st.caption(
            "This chart compares the component scores. "
            "Their contributions to the final score are "
            "weighted 60%, 25%, and 15%, respectively."
        )

        with st.expander("How is the final score calculated?"):

            st.write(
                f"Skill contribution: "
                f"{result['skill_score']:.2f} × 0.60 = "
                f"{result['skill_score'] * 0.60:.2f}"
            )

            st.write(
                f"Similarity contribution: "
                f"{result['similarity_score']:.2f} × 0.25 = "
                f"{result['similarity_score'] * 0.25:.2f}"
            )

            st.write(
                f"Completeness contribution: "
                f"{result['completeness_score']:.2f} × 0.15 = "
                f"{result['completeness_score'] * 0.15:.2f}"
            )

            st.write(
                f"**Final score: "
                f"{result['final_score']:.2f}/100**"
            )

    # ========================================
    # SKILLS AND COMPLETENESS TAB
    # ========================================

    with skills_tab:

        st.subheader("Job Description Skill Requirements")

        st.caption(
            "Requirements source: "
            + result["requirements_source"]
        )

        if result["required_skills"]:

            st.write(
                ", ".join(result["required_skills"])
            )

        else:

            st.info(
                "No recognizable skills were detected."
            )

        matched_col, missing_col = st.columns(2)

        with matched_col:

            st.markdown("#### ✅ Matched Skills")

            if result["matched_skills"]:

                for skill in result["matched_skills"]:

                    st.write(f"✓ {skill}")

            else:

                st.write("No required skills detected.")

        with missing_col:

            st.markdown("#### ⚪ Skills Not Detected")

            if result["missing_skills"]:

                for skill in result["missing_skills"]:

                    st.write(f"• {skill}")

            else:

                st.write(
                    "All listed requirements were detected."
                )

        st.caption(
            "A skill not detected in the text may still be "
            "known by the candidate. The system checks for "
            "known textual skill terms."
        )

        st.divider()

        st.subheader("Resume Completeness")

        all_sections = [
            "Skills",
            "Education",
            "Experience",
            "Projects",
            "Certifications",
            "Contact",
        ]

        found_sections = result["found_sections"]

        detected_col, missing_section_col = st.columns(2)

        with detected_col:

            st.markdown("#### ✅ Detected Indicators")

            if found_sections:

                for section in found_sections:

                    st.write(f"✓ {section}")

            else:

                st.write(
                    "No tracked indicators were detected."
                )

        with missing_section_col:

            st.markdown("#### ⚪ Not Detected")

            missing_sections = [
                section
                for section in all_sections
                if section not in found_sections
            ]

            if missing_sections:

                for section in missing_sections:

                    st.write(f"• {section}")

            else:

                st.write(
                    "All tracked indicators were detected."
                )

        st.caption(
            "Completeness uses keyword and pattern checks; "
            "it does not verify the quality or completeness "
            "of each section."
        )

    # ========================================
    # ROLE MATCHES TAB
    # ========================================

    with roles_tab:

        st.subheader("Suitable Roles from the Knowledge Base")

        st.write(
            "These rankings compare the resume's detected "
            "skills with each role's required skills."
        )

        role_rows = []

        for role in result["role_results"]:

            role_rows.append(
                {
                    "Role": role["role"],
                    "Skill Match": (
                        f"{role['score']:.2f}%"
                    ),
                    "Matched Skills": (
                        ", ".join(role["matched_skills"])
                        or "None detected"
                    ),
                    "Missing Skills": (
                        ", ".join(role["missing_skills"])
                        or "None"
                    ),
                }
            )

        role_df = pd.DataFrame(role_rows)

        st.dataframe(
            role_df,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "These are skill-overlap scores, not predicted "
            "hiring outcomes or validated suitability ratings."
        )

    # ========================================
    # EXTRACTED RESUME TAB
    # ========================================

    with text_tab:

        st.subheader("Extracted PDF Text")

        st.write(
            "Inspect the text used by the classification, "
            "skill matching, similarity, and completeness "
            "components."
        )

        st.text_area(
            "Resume text",
            value=result["resume_text"],
            height=450,
            disabled=True,
        )

    # ========================================
    # FINAL DISCLAIMER
    # ========================================

    st.divider()

    st.caption(
        "Academic project prototype • Uses a trained ML "
        "classifier, a predefined skill knowledge base, "
        "TF-IDF cosine similarity, and heuristic completeness "
        "checks. Results should be reviewed by a human. "
        "No score guarantees candidate quality or hiring success."
    )
