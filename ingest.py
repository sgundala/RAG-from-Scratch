import fitz  # PyMuPDF
from tqdm.auto import tqdm
import re

def text_formatter(text: str) -> str:
    if not text:
        return ""
    # keep it simple: flatten whitespace
    text = text.replace("\r", "\n").replace("\t", " ")
    text = " ".join(text.split()).strip()
    return text

def open_and_read_pdf(pdf_path: str, skip_first_pages: int = 41):
    """
    Returns a list of dicts, one per kept page:
      {
        "Page_number": <int>,                 # starts at 1 after skipping
        "Page_char_count": <int>,
        "page_word_count": <int>,
        "page_sentence_count_raw": <int>,     # split on . ! ?
        "page_token_count": <int>,            # ~heuristic: 1 token ≈ 4 chars
        "text": <str>
      }
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []

    # enumerate from 1 to match human page numbers
    for i, page in tqdm(enumerate(doc, start=1), total=len(doc), desc="Reading pages"):
        # skip front matter
        if i <= skip_first_pages:
            continue

        text = page.get_text("text")
        text = text_formatter(text)

        # simple counts (fast, no extra deps)
        char_count = len(text)
        word_count = len(text.split())
        sent_count = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
        token_count = char_count // 4  # heuristic; swap for tiktoken if you like

        pages_and_texts.append({
            "Page_number": i - skip_first_pages,     # adjusted: first kept page -> 1
            "Page_char_count": char_count,
            "page_word_count": word_count,
            "page_sentence_count_raw": sent_count,
            "page_token_count": token_count,
            "text": text
        })

    doc.close()
    return pages_and_texts

# ---- run it ----
pdf_path = "/content/Human-Nutrition.pdf"
pages_and_texts = open_and_read_pdf(pdf_path, skip_first_pages=41)

# preview first 2 kept pages
pages_and_texts[:2]
