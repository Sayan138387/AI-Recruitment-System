# AI-Based Recruitment and Resume Screening System

An AI-assisted recruitment platform that extracts resume text, classifies resumes, matches candidates with job roles, analyzes job-description compatibility, and calculates a project-specific Applicant Tracking System (ATS) score.

The project combines machine learning, natural language processing, resume parsing, skill matching, and an interactive Streamlit dashboard.

## Features

### 1. Resume Upload and Text Extraction
- Upload PDF resumes through the Streamlit dashboard.
- Extract selectable text from regular PDFs using `pdfplumber`.
- Use Tesseract OCR for scanned, image-based PDF pages.
- Render scanned pages with PyMuPDF and preprocess images using Pillow.
- Display the extracted resume text for inspection.

### 2. Machine Learning Resume Classification
- Use TF-IDF features and Logistic Regression for resume classification.
- Compare a neural-network classifier based on `MLPClassifier`.
- Display predicted resume categories and model outputs.

The Logistic Regression model achieved approximately 65.4% accuracy on the project's recorded test split. This is an experimental result for the current dataset, not a guarantee of performance on unseen real-world resumes.

### 3. Job Role Matching
- Maintain role and skill information in a JSON knowledge base.
- Compare resume skills with the requirements of supported roles.
- Identify matching skills and potential missing skills.
- Help users explore relevant job roles.

### 4. ATS Score Analysis
The system calculates a project-specific ATS score using three components:

- **Skill Match — 60%:** Percentage of required skills identified in the resume.
- **Job Description Similarity — 25%:** Text similarity calculated using TF-IDF and cosine similarity.
- **Resume Completeness — 15%:** Presence of selected resume sections, such as skills, education, experience, projects, certifications, and contact information.

The final score is calculated as:

`ATS Score = (Skill Match × 0.60) + (Job Similarity × 0.25) + (Completeness × 0.15)`

All three component scores are expressed as percentages.

**Important:** This is a project-defined heuristic, not an industry-standard ATS formula or a prediction of whether an employer will hire a candidate. OCR errors and resume formatting can affect the results.

### 5. Additional Analysis Modules
The repository also includes modules for:

- TF-IDF-based text search and cosine similarity.
- Historical job-posting and IT job-demand analysis.
- Time-series analysis and visualizations.
- Knowledge-base management and role matching.
- Model training and evaluation.

The availability and output of these modules depend on their required datasets and model artifacts.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| User interface | Streamlit |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| NLP and text similarity | TF-IDF, cosine similarity |
| PDF extraction | pdfplumber, PyMuPDF |
| OCR | Tesseract OCR, pytesseract |
| Image processing | Pillow |
| Data visualization | Matplotlib |
| Model persistence | Joblib |
| Configuration and knowledge base | JSON |

## Project Structure

```text
AI-Recruitment-System/
├── app.py
├── requirements.txt
├── README.md
├── datasets/
│   └── job_skill_knowledge.json
├── models/
│   ├── resume_classifier.pkl
│   └── tfidf_vectorizer.pkl
├── resumes/
├── src/
│   ├── ats_score.py
│   ├── deep_learning.py
│   ├── inspect_jobs.py
│   ├── inspect_time_series.py
│   ├── job_demand.py
│   ├── knowledge_base.py
│   ├── pdf_parser.py
│   ├── predict_resume.py
│   ├── role_matcher.py
│   ├── search_engine.py
│   ├── time_series.py
│   └── train_final_model.py
└── venv/
```

This structure is representative of the main project files. Additional scripts and data files may also be present. The `venv/` directory is generated locally and should not be committed to Git.

## Requirements

- Python installed on your computer.
- Git, if cloning the repository.
- Tesseract OCR installed separately for scanned-PDF support.
- The trained model files and datasets required by the modules you intend to run.

Python libraries are listed in `requirements.txt`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sayan138387/AI-Recruitment-System.git
cd AI-Recruitment-System
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the appropriate Python virtual-environment activation method permitted by your system.

### 3. Install Python dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Install Tesseract OCR

Install the Tesseract OCR application separately on Windows.

The current PDF parser checks the system PATH and these common locations:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
C:\Program Files (x86)\Tesseract-OCR\tesseract.exe
```

Verify the installation:

```powershell
tesseract --version
```

If Tesseract is installed in a custom location, add its folder to PATH or update the executable lookup in `src/pdf_parser.py`.

### 5. Verify model artifacts

Before starting the dashboard, verify that the trained model files expected by the application are available:

```text
models/resume_classifier.pkl
models/tfidf_vectorizer.pkl
```

These files may not be included in a fresh Git clone because generated model artifacts are excluded by the repository's current Git ignore configuration. Obtain the appropriate artifacts through your project's approved sharing method or generate them using the training workflow and required datasets.

Do not commit large datasets, private resumes, or unnecessary generated files.

## Run the Application

From the project root, with the virtual environment activated:

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Using the Dashboard

1. Upload a PDF resume.
2. Select the target job role.
3. Enter the relevant job description.
4. Submit the analysis.
5. Review the extracted resume text, predicted category, matching roles, matched and missing skills, completeness indicators, and ATS score.
6. Download the available analysis report when needed.

For scanned resumes, extraction quality depends on the image resolution, document layout, language, and OCR accuracy.

## Testing

The project has undergone the following development checks:

- Python syntax checks across 20 Python files.
- Python imports for the OCR dependencies.
- Text extraction from a regular PDF.
- OCR extraction from an image-only test PDF.
- Resume analysis through the Streamlit dashboard.

These checks confirm the tested functionality in the development environment. They do not constitute exhaustive testing across all PDFs, operating systems, datasets, or deployment environments.

## Limitations

- Resume classification and similarity scores depend on the current dataset and model configuration.
- OCR can misread characters, email addresses, technologies, and text arranged in multiple columns.
- A keyword-based skill matcher may miss synonyms or related experience.
- The ATS score is an experimental heuristic and should not be treated as an authoritative hiring decision.
- Some modules require datasets or model files that may not be included in the public repository.
- Model training results can vary with dataset quality, preprocessing, and the train-test split.

## Future Improvements

- Improve OCR accuracy and resume-layout handling.
- Expand skill synonyms and normalize technology names.
- Add automated tests for parsing, matching, and scoring.
- Make model training reproducible from a clean environment.
- Improve PDF report generation and dashboard error handling.
- Evaluate classification and scoring on a broader, representative dataset.
- Document module-specific dataset requirements and evaluation metrics.

## Author

**Sayan**

GitHub: https://github.com/Sayan138387

## Disclaimer

This project is intended for academic and experimental purposes. Its predictions and scores are informational aids, not substitutes for human review, and should not be used as the sole basis for employment decisions.
