import os

import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import easyocr
import tempfile
import torch


class PDFConverter:
    def __init__(self):
        # Check GPU availability
        gpu = True if torch.cuda.is_available() else False
        if gpu:
            print("✅ Sử dụng GPU để xử lý")
            print(f"\n- CUDA available: {torch.cuda.is_available()}")
            print(f"\n- CUDA version: {torch.version.cuda}")
            print(f"\n- GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ Không tìm thấy GPU, sử dụng CPU")

        # Initialize EasyOCR with Vietnamese language support
        self.reader = easyocr.Reader(["vi"], gpu=gpu)
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "output"
        )

    def pdf_to_text(self, pdf_path, output_format="txt"):
        """
        Convert PDF to text using PyMuPDF and EasyOCR with Vietnamese support
        """
        if not os.path.exists(pdf_path):
            print(f"PDF file not found: {pdf_path}")
            return None

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            all_text = []
            print(f"🔍 Đang chuyển đổi {pdf_path}...")

            for page_num in range(len(pdf_document)):
                # Get page
                page = pdf_document[page_num]

                # Convert page to image with higher DPI for better OCR
                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))

                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Convert PIL Image to numpy array for EasyOCR
                img_array = np.array(img)

                # Perform OCR with EasyOCR
                results = self.reader.readtext(img_array)

                # Extract text from results
                page_text = "\n".join([text[1] for text in results])
                all_text.append(page_text)

            # Create output filename
            pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            os.makedirs(self.output_dir, exist_ok=True)

            if output_format == "txt":
                output_path = os.path.join(self.output_dir, f"{pdf_filename}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("\n\n".join(all_text))
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            pdf_document.close()
            return output_path

        except Exception as e:
            print(f"❌ Lỗi chuyển đổi PDF: {e}")
            return None

    def cleanup(self):
        import shutil

        try:
            if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
