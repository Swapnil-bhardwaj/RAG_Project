import os
import re
import PyPDF2
from bs4 import BeautifulSoup

def _clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_pdf(path: str) -> str:
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            text.append(txt)
    return _clean_text("\n".join(text))

def parse_markdown(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    raw = re.sub(r"```.*?```", " ", raw, flags=re.S)
    raw = re.sub(r"[#>*_`~\[\]()-]", " ", raw)
    return _clean_text(raw)

def parse_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return _clean_text(soup.get_text(separator=" "))

def parse_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext in {".md", ".markdown"}:
        return parse_markdown(path)
    if ext in {".html", ".htm"}:
        return parse_html(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _clean_text(f.read())
