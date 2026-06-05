from pathlib import Path
import fitz
import re


######## Functions for text extraction from PDF and TXT files ########
    
def extract_text_from_pdf(file_path):

    """
    Extract text from a PDF file using PyMuPDF.
    """
    text_pages = []

    with fitz.open(file_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            page_text = page.get_text("text")

            if page_text.strip():
                text_pages.append(
                    f"\n--- Page {page_number} ---\n{page_text.strip()}"
                )

    extracted_text = "\n".join(text_pages).strip()

    if not extracted_text:
        raise ValueError(f"No text could be extracted from PDF: {file_path}")

    return extracted_text
    
def extract_text_from_txt(file_path: str) -> str:
    path = Path(file_path)
    text_pages = []

    with path.open('r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                text_pages.append(line.strip())

    extracted_text = "\n".join(text_pages).strip()

    return extracted_text



def extract_text_from_file(file_path):

    """
    Extract raw text from a PDF/text file using PyMuPDF.
    """

    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.lower().endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(('.txt', '.md')):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    

###  Function for research Agent to get name and github profile of candidate##
def extract_candidate_name(resume_text: str) -> str:
    lines = [line.strip("# ").strip() for line in resume_text.splitlines() if line.strip()]
    return lines[0] if lines else "candidate"

def extract_email_from_text(text: str) -> str | None:
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    match = re.search(pattern, text)

    if match:
        return match.group(0)

    return None


def extract_github_url(resume_text: str) -> str | None:
    pattern = r"(https?://)?(www\.)?github\.com/[A-Za-z0-9_.-]+"
    match = re.search(pattern, resume_text)

    if not match:
        return None

    github_url = match.group(0)

    if not github_url.startswith("http"):
        github_url = "https://" + github_url

    return github_url

