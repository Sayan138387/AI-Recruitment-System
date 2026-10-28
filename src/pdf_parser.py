
import io
import shutil
from pathlib import Path

import pdfplumber
import pymupdf
import pytesseract
from PIL import Image, ImageOps


# ============================================================
# 1. CONFIGURE TESSERACT OCR
# ============================================================

def configure_tesseract():
    """
    Find the Tesseract OCR executable.

    First checks PATH, then checks common Windows
    installation directories.
    """

    tesseract_path = shutil.which("tesseract")

    if not tesseract_path:
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]

        for path in possible_paths:
            if Path(path).is_file():
                tesseract_path = path
                break

    if not tesseract_path:
        raise FileNotFoundError(
            "Tesseract OCR was not found. "
            "Please install Tesseract OCR or add it to PATH."
        )

    pytesseract.pytesseract.tesseract_cmd = tesseract_path


# ============================================================
# 2. READ PDF INPUT
# ============================================================

def get_pdf_bytes(pdf_source):
    """
    Accept a PDF from any of these sources:
    - File path
    - Bytes or bytearray
    - Streamlit UploadedFile
    - BytesIO or another file-like object
    """

    if isinstance(pdf_source, (str, Path)):
        data = Path(pdf_source).read_bytes()

    elif isinstance(
        pdf_source,
        (bytes, bytearray, memoryview),
    ):
        data = bytes(pdf_source)

    elif hasattr(pdf_source, "getvalue"):
        # Supports Streamlit UploadedFile and BytesIO.
        data = pdf_source.getvalue()

    elif hasattr(pdf_source, "read"):
        # Supports generic file-like objects.
        try:
            pdf_source.seek(0)
        except (AttributeError, OSError):
            pass

        data = pdf_source.read()

        try:
            pdf_source.seek(0)
        except (AttributeError, OSError):
            pass

    else:
        raise TypeError(
            "Unsupported input. Provide a PDF path, "
            "PDF bytes, or a file-like object."
        )

    if not data:
        raise ValueError("The supplied PDF file is empty.")

    return data


# ============================================================
# 3. EXTRACT TEXT USING OCR
# ============================================================

def extract_text_with_ocr(pdf_page):
    """
    Render a PDF page as an image, preprocess it,
    and extract text using Tesseract OCR.
    """

    configure_tesseract()

    # Render at 3x scale to provide a higher-resolution image.
    pixmap = pdf_page.get_pixmap(
        matrix=pymupdf.Matrix(3, 3),
        alpha=False,
    )

    image_bytes = pixmap.tobytes("png")

    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()

        # Convert to grayscale.
        processed_image = image.convert("L")

        # Improve contrast while retaining grayscale details.
        processed_image = ImageOps.autocontrast(
            processed_image
        )

        # OCR configuration:
        # OEM 3 = use the available modern OCR engine.
        # PSM 3 = automatically detect page layout.
        extracted_text = pytesseract.image_to_string(
            processed_image,
            lang="eng",
            config="--oem 3 --psm 3",
        )

    return extracted_text.strip()


# ============================================================
# 4. MAIN PDF TEXT EXTRACTION FUNCTION
# ============================================================

def extract_text_from_pdf(pdf_source):
    """
    Extract text from normal and scanned PDF documents.

    Workflow:
    1. Try normal text extraction using pdfplumber.
    2. If a page contains very little extractable text,
       use Tesseract OCR as a fallback.
    3. Keep the better available text for that page.
    4. Combine all pages into one string.

    The function supports both file paths and Streamlit uploads.
    """

    pdf_bytes = get_pdf_bytes(pdf_source)
    extracted_pages = []

    # Open the PDF for selectable-text extraction and rendering.
    with (
        pdfplumber.open(io.BytesIO(pdf_bytes)) as text_pdf,
        pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf",
        ) as render_pdf,
    ):

        for page_number, page in enumerate(text_pdf.pages):

            # First attempt: extract selectable PDF text.
            page_text = (
                page.extract_text() or ""
            ).strip()

            # A very short result may indicate a scanned page.
            # OCR is only attempted when needed.
            if len(page_text) < 80:

                rendered_page = render_pdf[page_number]

                ocr_text = extract_text_with_ocr(
                    rendered_page
                )

                # Prefer OCR when it returns more text.
                # Otherwise, preserve the original extraction.
                if len(ocr_text) > len(page_text):
                    page_text = ocr_text

            if page_text:
                extracted_pages.append(page_text)

    return "\n\n".join(extracted_pages)


# ============================================================
# 5. TEST THE PDF PARSER DIRECTLY
# ============================================================

if __name__ == "__main__":

    pdf_path = input(
        "Enter the path to a PDF resume: "
    ).strip().strip('"')

    try:
        extracted_text = extract_text_from_pdf(pdf_path)

        print("\n" + "=" * 60)
        print("EXTRACTED RESUME TEXT")
        print("=" * 60 + "\n")

        if extracted_text:
            print(extracted_text)
        else:
            print(
                "No text could be extracted. "
                "The PDF may be blank or difficult to read."
            )

        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)

    except Exception as error:
        print(f"\nPDF extraction failed: {error}")
