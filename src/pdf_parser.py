import pdfplumber


def extract_text_from_pdf(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


if __name__ == "__main__":

    pdf_path = "resumes/test_resume.pdf"

    text = extract_text_from_pdf(pdf_path)

    print("\nExtracted Resume Text:\n")
    print(text)