import csv
import pymupdf  # PyMuPDF for PDFs
import docx  # python-docx for Word files
import openpyxl  # openpyxl for Excel files
from pptx import Presentation  # python-pptx for PowerPoint files

# Every parser returns a list of pages like this:
#   [{"page": "Page 1", "text": "..."}, {"page": "Page 2", "text": "..."}]
# Keeping the page label lets us cite the exact page in answers.


def parse_file(file_path):
    path = file_path.lower()
    if path.endswith(".pdf"):
        return parse_pdf(file_path)
    elif path.endswith(".docx"):
        return parse_docx(file_path)
    elif path.endswith(".pptx"):
        return parse_pptx(file_path)
    elif path.endswith(".xlsx"):
        return parse_xlsx(file_path)
    elif path.endswith(".csv"):
        return parse_csv(file_path)
    elif path.endswith(".txt"):
        return parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def parse_pdf(file_path):
    pages = []
    doc = pymupdf.open(file_path)
    for number, page in enumerate(doc, start=1):
        pages.append({"page": f"Page {number}", "text": page.get_text()})
    return pages


def parse_docx(file_path):
    doc = docx.Document(file_path)
    lines = [p.text for p in doc.paragraphs]
    # Also read text inside tables
    for table in doc.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text for cell in row.cells))
    # Word files do not have fixed pages, so we keep one section
    return [{"page": "Document", "text": "\n".join(lines)}]


def parse_pptx(file_path):
    prs = Presentation(file_path)
    pages = []
    for number, slide in enumerate(prs.slides, start=1):
        text = ""
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
        pages.append({"page": f"Slide {number}", "text": text})
    return pages


def parse_xlsx(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    pages = []
    # One page per sheet
    for sheet in wb.worksheets:
        text = ""
        for row in sheet.iter_rows(values_only=True):
            text += "\t".join("" if cell is None else str(cell) for cell in row) + "\n"
        pages.append({"page": f"Sheet {sheet.title}", "text": text})
    return pages


def parse_csv(file_path):
    text = ""
    with open(file_path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            text += ",".join(row) + "\n"
    return [{"page": "Table", "text": text}]


def parse_txt(file_path):
    with open(file_path, encoding="utf-8", errors="ignore") as f:
        return [{"page": "Text", "text": f.read()}]
