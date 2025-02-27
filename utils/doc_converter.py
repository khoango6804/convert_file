import win32com.client
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
import mimetypes
import zipfile

def validate_doc_file(file_path):
    """Check if file is a valid Word document"""
    # For .docx (which is actually a zip archive)
    if file_path.endswith('.docx'):
        try:
            with zipfile.ZipFile(file_path) as z:
                content_types = '[Content_Types].xml'
                if content_types not in z.namelist():
                    return False
                
                # Check for document.xml which should exist in valid Word files
                if not any('word/document.xml' in name for name in z.namelist()):
                    return False
                return True
        except zipfile.BadZipFile:
            return False
    
    # For .doc files, we'll check file signature/magic bytes
    elif file_path.endswith('.doc'):
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                # DOC file signature: D0 CF 11 E0 A1 B1 1A E1
                if header.startswith(b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1'):
                    return True
                return False
        except:
            return False
            
    return False


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
        # Validate file first
        if not validate_doc_file(input_path):
            print(f"❌ Invalid Word document format: {input_path}")
            # Try alternative extraction method for problematic files
            return try_alternative_extraction(input_path)
            
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        
        # Add timeout handling
        word.DisplayAlerts = False
        
        # Convert to docx
        doc = word.Documents.Open(input_path)
        docx_path = input_path + "x"
        doc.SaveAs2(docx_path, FileFormat=16)  # 16 = docx format
        doc.Close()
        word.Quit()

        # Process the docx file
        return extract_text_from_docx(docx_path)

    except Exception as e:
        print(f"❌ Lỗi mở file {input_path}: {str(e)}")
        # Try alternative method when exception occurs
        return try_alternative_extraction(input_path)


def identify_office_file_type(file_path):
    """Identify actual Office file type regardless of extension"""
    try:
        with zipfile.ZipFile(file_path) as z:
            # Check content types
            if any('theme/theme' in name for name in z.namelist()):
                return "Office Theme File"
            elif any('word/document.xml' in name for name in z.namelist()):
                return "Word Document"
            elif any('xl/workbook.xml' in name for name in z.namelist()):
                return "Excel Workbook"
            elif any('ppt/presentation.xml' in name for name in z.namelist()):
                return "PowerPoint Presentation"
            return "Unknown Office Format"
    except:
        return "Not an Office Open XML file"


def try_alternative_extraction(input_path):
    """Try alternative methods to extract text from problematic files"""
    
    # First, let's identify what the file actually is
    if input_path.endswith('.docx'):
        file_type = identify_office_file_type(input_path)
        if file_type != "Word Document":
            print(f"⚠️ File mismatch: {input_path} is actually a {file_type}")
            
            # For theme files, we can try to extract any text content
            if file_type == "Office Theme File":
                try:
                    with zipfile.ZipFile(input_path) as z:
                        # Look for XML files that might contain text
                        xml_files = [name for name in z.namelist() if name.endswith('.xml')]
                        content = []
                        for xml_file in xml_files:
                            try:
                                text = z.read(xml_file).decode('utf-8', errors='ignore')
                                # Extract text between tags
                                import re
                                extracted = re.findall(r'>([^<]+)<', text)
                                content.extend([t.strip() for t in extracted if t.strip()])
                            except:
                                pass
                                
                        if content:
                            return "\n\n".join(content)
                except:
                    pass
    
    # Try other methods for any file type
    try:
        # Try using textract library if available
        try:
            import textract
            text = textract.process(input_path).decode('utf-8')
            if text.strip():
                return text
        except:
            pass

        # Try antiword for .doc files
        if input_path.endswith('.doc'):
            try:
                import subprocess
                result = subprocess.run(['antiword', input_path], capture_output=True, text=True)
                if result.stdout.strip():
                    return result.stdout
            except:
                pass
                
        # All methods failed
        return None
    except Exception as e:
        print(f"❌ All extraction methods failed for {input_path}: {str(e)}")
        return None
