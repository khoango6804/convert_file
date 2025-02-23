import win32com.client
from docx import Document

def extract_text_from_doc(input_path):
    """Mở file .doc trong Microsoft Word, copy nội dung và trả về văn bản"""
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False  # Chạy Word ẩn
        doc = word.Documents.Open(input_path)
        text = doc.Content.Text
        doc.Close()
        word.Quit()
        return text.strip()
    except Exception as e:
        print(f"❌ Lỗi mở file {input_path}: {e}")
        return None

def extract_text_from_docx(input_path):
    """Mở file .docx bằng python-docx, copy nội dung và trả về văn bản"""
    try:
        doc = Document(input_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"❌ Lỗi mở file {input_path}: {e}")
        return None
