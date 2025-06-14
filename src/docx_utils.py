from typing import List, Dict
from docx import Document
from .config import MIN_WORDS_PARAGRAPH

def extract_paragraphs_with_type(docx_path) -> List[Dict[str, str]]:
    """
    Returns a list of paragraph objects, each a dict with 'text' and 'type'.
    'type' is either 'content' or 'heading'.
    This preserves the original document order.

    * 'content': Paragraphs with ≥ MIN_WORDS_PARAGRAPH words.
    * 'heading': Paragraphs with < MIN_WORDS_PARAGRAPH words.
    """
    doc = Document(docx_path)
    paragraphs = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        
        para_type = 'content' if len(text.split()) >= MIN_WORDS_PARAGRAPH else 'heading'
        paragraphs.append({'text': text, 'type': para_type})
        
    return paragraphs
