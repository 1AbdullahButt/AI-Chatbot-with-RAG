import csv
import pymupdf  # PyMuPDF for PDFs
import docx  # python-docx for Word files
import openpyxl  # openpyxl for Excel files
from pptx import Presentation  # python-pptx for PowerPoint files


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
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def parse_pdf(file_path):
    doc = pymupdf.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text


def parse_docx(file_path):
    doc = docx.Document(file_path)
    lines = [p.text for p in doc.paragraphs]
    # Also read text inside tables
    for table in doc.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text for cell in row.cells))
    return "\n".join(lines)


def parse_pptx(file_path):
    prs = Presentation(file_path)
    text = ""
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
    return text


def parse_xlsx(file_path):
    wb = openpyxl.load_workbook(file_path, data_only=True)
    text = ""
    # Read every sheet, not just the first one
    for sheet in wb.worksheets:
        text += f"Sheet: {sheet.title}\n"
        for row in sheet.iter_rows(values_only=True):
            text += "\t".join("" if cell is None else str(cell) for cell in row) + "\n"
    return text


def parse_csv(file_path):
    text = ""
    with open(file_path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            text += ",".join(row) + "\n"
    return text
