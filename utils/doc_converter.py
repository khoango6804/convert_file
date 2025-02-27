import win32com.client
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH


def convert_alignment_to_markdown(paragraph):
    """Convert Word alignment to Markdown format"""
    if paragraph.alignment == WD_ALIGN_PARAGRAPH.CENTER:
        return f"{paragraph.text}  "  # Two spaces for center alignment
    elif paragraph.alignment == WD_ALIGN_PARAGRAPH.RIGHT:
        return f"    {paragraph.text}"  # Four spaces for right alignment
    return paragraph.text


def convert_style_to_markdown(paragraph):
    """Convert Word text style to Markdown"""
    text = paragraph.text.strip()
    if not text:
        return ""

    # Check for heading styles
    if paragraph.style.name.startswith("Heading"):
        level = int(paragraph.style.name[-1])
        return f"{'#' * level} {text}"

    # Check for bold/italic
    if any(run.bold for run in paragraph.runs):
        if "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" in text:
            text = f"**{text}**"
    if any(run.italic for run in paragraph.runs):
        if "Độc lập - Tự do - Hạnh phúc" in text or "ngày" in text.lower():
            text = f"*{text}*"

    return text


def extract_text_from_docx(input_path):
    """Extract content from .docx and convert to Markdown"""
    try:
        doc = Document(input_path)
        markdown_lines = []
        prev_empty = True

        for paragraph in doc.paragraphs:
            # Convert paragraph to markdown
            md_text = convert_style_to_markdown(paragraph)
            if md_text:
                # Handle alignment
                md_text = convert_alignment_to_markdown(paragraph)

                # Handle lists
                if paragraph.style.name.startswith("List"):
                    md_text = f"- {md_text}"

                # Add blank lines around headings and between paragraphs
                if md_text.startswith("#") or not prev_empty:
                    markdown_lines.append("")

                markdown_lines.append(md_text)
                prev_empty = False
            else:
                if not prev_empty:
                    markdown_lines.append("")
                prev_empty = True

        # Join with newlines and ensure single newline at end
        return "\n".join(markdown_lines).strip() + "\n"

    except Exception as e:
        print(f"❌ Lỗi mở file {input_path}: {e}")
        return None


def extract_text_from_doc(input_path):
    """Convert .doc to .docx then extract content"""
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False

        # Convert to docx
        doc = word.Documents.Open(input_path)
        docx_path = input_path + "x"
        doc.SaveAs2(docx_path, FileFormat=16)  # 16 = docx format
        doc.Close()
        word.Quit()

        # Process the docx file
        return extract_text_from_docx(docx_path)

    except Exception as e:
        print(f"❌ Lỗi mở file {input_path}: {e}")
        return None
