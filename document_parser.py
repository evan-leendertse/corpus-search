import os
import subprocess
import tempfile
from pathlib import Path
import PyPDF2
import pdfplumber
from docx import Document

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyPDF2."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def convert_pages_to_pdf(pages_path):
    """Convert Apple Pages file to PDF using textutil."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_pdf = tmp.name
        
        # Use textutil to convert Pages to PDF
        cmd = ['textutil', '-convert', 'pdf', '-output', tmp_pdf, str(pages_path)]
        subprocess.run(cmd, check=True, capture_output=True)
        
        return tmp_pdf
    except subprocess.CalledProcessError as e:
        print(f"Error converting {pages_path} to PDF: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error converting {pages_path}: {e}")
        return None

def extract_text_from_pages(pages_path):
    """Extract text from Pages file by converting to PDF first."""
    pdf_path = convert_pages_to_pdf(pages_path)
    if pdf_path and os.path.exists(pdf_path):
        try:
            text = extract_text_from_pdf(pdf_path)
            return text
        finally:
            os.unlink(pdf_path)  # Clean up temp PDF
    return ""

def parse_document(file_path):
    """Parse document based on extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix == '.pdf':
        return extract_text_from_pdf(path)
    elif suffix == '.pages':
        return extract_text_from_pages(path)
    else:
        print(f"Unsupported file format: {suffix}")
        return ""

def batch_parse_documents(directory, extensions=['.pdf']):
    """Parse all documents in directory with given extensions."""
    documents = {}
    directory = Path(directory)
    
    for ext in extensions:
        for file_path in directory.rglob(f'*{ext}'):
            try:
                text = parse_document(file_path)
                if text.strip():
                    documents[str(file_path)] = text
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
    
    return documents