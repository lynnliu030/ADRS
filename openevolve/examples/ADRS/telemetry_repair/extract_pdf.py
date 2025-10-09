#!/usr/bin/env python3
"""
Extract text from PDF research papers for OpenEvolve context
"""

try:
    import pdfplumber
    PDF_LIBRARY = "pdfplumber"
except ImportError:
    try:
        import PyPDF2
        PDF_LIBRARY = "PyPDF2"
    except ImportError:
        print("Please install: pip install pdfplumber PyPDF2")
        exit(1)

def extract_text_pdfplumber(pdf_path: str) -> str:
    """Extract text using pdfplumber (better for academic papers)"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text

def extract_text_pypdf2(pdf_path: str) -> str:
    """Extract text using PyPDF2 (fallback option)"""
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
    return text

def clean_extracted_text(text: str) -> str:
    """Clean up extracted text for better LLM consumption"""
    # Remove excessive whitespace
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]  # Remove empty lines
    
    # Join with single newlines
    cleaned = '\n'.join(lines)
    
    # Remove page numbers and common artifacts
    import re
    cleaned = re.sub(r'\n\d+\n', '\n', cleaned)  # Remove standalone page numbers
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Max 2 consecutive newlines
    
    return cleaned

def extract_pdf_to_markdown(pdf_path: str, output_path: str = "research_context.md"):
    """Extract PDF and save as formatted markdown"""
    
    print(f"📖 Extracting text from {pdf_path} using {PDF_LIBRARY}...")
    
    # Extract text
    if PDF_LIBRARY == "pdfplumber":
        raw_text = extract_text_pdfplumber(pdf_path)
    else:
        raw_text = extract_text_pypdf2(pdf_path)
    
    # Clean text
    cleaned_text = clean_extracted_text(raw_text)
    
    # Format as markdown
    markdown_content = f"""# Research Context: Extracted from {pdf_path}

## Full Paper Content

{cleaned_text}

---
*Extracted automatically. You may want to edit this content to highlight the most relevant sections for network telemetry repair.*
"""
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"✅ Extracted {len(cleaned_text)} characters to {output_path}")
    print(f"💡 Review and edit {output_path} to highlight relevant sections")
    
    return cleaned_text

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract_pdf.py <path_to_paper.pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    extract_pdf_to_markdown(pdf_path) 