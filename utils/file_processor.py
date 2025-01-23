from PyPDF2 import PdfReader

def process_documents(files):
    """Reads PDF files and extracts text."""
    texts = []
    for file in files:
        pdf_reader = PdfReader(file)
        text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        texts.append(text)
    return texts
