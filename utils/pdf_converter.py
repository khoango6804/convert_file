import os
import fitz
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from google import genai
import tempfile
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential


class PDFConverter:
    def __init__(self):
        # Load API key from config
        try:
            with open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "api.json"),
                "r",
            ) as f:
                config = json.load(f)
                api_key = config.get("api_key")
                if not api_key:
                    raise ValueError("API key not found in config")

                # Configure Gemini
                self.client = genai.Client(api_key=api_key)
                print("✅ Đã kết nối Gemini API")
        except Exception as e:
            print(f"❌ Lỗi cấu hình Gemini API: {e}")
            raise

        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "output"
        )
        self.retry_count = 0
        self.max_retries = 3
        self.wait_time = 10  # seconds

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_gemini_api(self, prompt, image=None):
        """Call Gemini API with retry logic"""
        try:
            if image:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash", contents=[prompt, image]
                )
            else:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash", contents=[prompt]
                )
            return response

        except Exception as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                print(f"⚠️ API đang bị giới hạn. Thử lại sau {self.wait_time} giây...")
                time.sleep(self.wait_time)
                self.retry_count += 1
                if self.retry_count >= self.max_retries:
                    raise Exception("Đã vượt quá số lần thử lại API")
                raise  # Retry
            raise  # Other errors

    def preprocess_image(self, image):
        """
        Preprocess image to improve OCR quality
        Args:
            image: PIL Image object
        Returns:
            Preprocessed PIL Image object
        """
        try:
            # 1. Convert to grayscale
            img_gray = ImageOps.grayscale(image)

            # 2. Increase contrast
            enhancer = ImageEnhance.Contrast(img_gray)
            img_contrast = enhancer.enhance(2.0)  # Increase contrast by factor of 2

            # 3. Binarization using Otsu's method
            img_array = np.array(img_contrast)
            threshold = int(np.mean(img_array))
            img_binary = Image.fromarray((img_array > threshold).astype(np.uint8) * 255)

            # 4. Denoise
            img_denoised = ImageOps.autocontrast(img_binary, cutoff=1)

            # 5. Remove borders
            bbox = ImageOps.invert(img_denoised).getbbox()
            if bbox:
                img_cropped = img_denoised.crop(bbox)
            else:
                img_cropped = img_denoised

            print("✨ Đã xử lý ảnh để tăng chất lượng OCR")
            return img_cropped

        except Exception as e:
            print(f"⚠️ Lỗi xử lý ảnh: {e}")
            return image

    def check_vietnamese_text(self, text: str) -> str:
        """
        Validate and correct Vietnamese text using Gemini
        """
        try:
            prompt = """
            Nhiệm vụ: Chuẩn hóa văn bản Markdown và sửa lỗi chính tả tiếng Việt.
            
            Yêu cầu định dạng Markdown:
            1. Cấu trúc tiêu đề:
            - Chỉ một tiêu đề cấp 1 (#) ở đầu văn bản
            - Sử dụng ## cho các tiêu đề phụ
            - Không lặp lại tiêu đề cấp 1

            2. Định dạng bảng:
            - Căn chỉnh cột bảng
            - Số liệu căn phải (:---)
            - Văn bản căn trái (:---)
            - Đảm bảo khoảng cách trong ô

            3. Định dạng văn bản:
            - Sửa các lỗi chính tả và dấu câu
            - Chuẩn hóa khoảng cách
            - Giữ nguyên các ký tự đặc biệt
            - Kết thúc file với một dòng trống

            4. Bảo toàn nội dung:
            - Giữ nguyên các mã số văn bản
            - Giữ nguyên các thuật ngữ chuyên ngành
            - Không thay đổi ý nghĩa văn bản
            - Không thêm các khối lệnh markdown

            Lưu ý:
            - Không bao gồm dấu ``` hoặc ```markdown
            - Không thêm thông tin không cần thiết
            - Đảm bảo văn bản kết thúc với một dòng trống
            """

            # Get response from Gemini
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", contents=[prompt, text]
            )

            return response.text if response.text else text

        except Exception as e:
            print(f"⚠️ Lỗi kiểm tra chính tả: {e}")
            return text

    def clean_markdown_content(self, text: str) -> str:
        """
        Clean markdown content and normalize spacing while preserving content structure
        """
        try:
            lines = text.split("\n")
            cleaned_lines = []
            prev_line_empty = False

            for line in lines:
                # Remove ``` markers
                if line.strip().startswith("```"):
                    line = line.replace("```markdown", "").replace("```", "").strip()

                # Skip if line is empty after cleaning
                if not line.strip():
                    if not prev_line_empty:  # Only add one empty line
                        cleaned_lines.append("")
                        prev_line_empty = True
                    continue

                # Handle headings (add space after #)
                if line.strip().startswith("#"):
                    line = line.strip()
                    parts = line.split(" ", 1)
                    if len(parts) > 1:
                        line = f"{parts[0]} {parts[1].strip()}"

                # Handle lists (preserve indentation)
                elif line.strip().startswith(("-", "*", "1.")):
                    indent = len(line) - len(line.lstrip())
                    line = " " * indent + line.strip()

                # Handle tables (normalize spacing)
                elif "|" in line:
                    cells = [cell.strip() for cell in line.split("|")]
                    line = "|".join(cells)

                # Normal lines
                else:
                    line = line.strip()

                cleaned_lines.append(line)
                prev_line_empty = False

            # Ensure single newline at end
            cleaned_text = "\n".join(cleaned_lines).strip() + "\n"
            return cleaned_text

        except Exception as e:
            print(f"⚠️ Lỗi khi làm sạch Markdown: {e}")
            return text

    def pdf_to_text(self, pdf_path, output_format="txt"):
        """
        Convert PDF to text using PyMuPDF and Gemini Vision
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

                # Convert page to image with higher DPI
                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Preprocess image
                img_processed = self.preprocess_image(img)

                # Create prompt for Gemini
                prompt = """
                Nhiệm vụ: Chuyển đổi văn bản thành định dạng Markdown chuẩn.

                Yêu cầu định dạng:
                1. Tiêu đề và cấu trúc:
                - # cho tiêu đề chính (tên cơ quan, tên văn bản)
                - ## cho tiêu đề cấp 2 (số văn bản, trích yếu)
                - ### cho các phần chính của văn bản
                - #### cho tiêu đề phụ

                2. Định dạng văn bản:
                - **text** cho văn bản in đậm
                - *text* cho văn bản in nghiêng
                - > cho trích dẫn và ghi chú
                - --- cho đường kẻ ngang phân cách

                3. Bảng và danh sách:
                - Sử dụng | và - cho bảng
                - Căn lề số liệu sang phải trong bảng
                - Sử dụng - hoặc * cho danh sách không thứ tự
                - Sử dụng 1. 2. 3. cho danh sách có thứ tự

                4. Đặc biệt với văn bản hành chính:
                - In đậm các cụm từ quan trọng như "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"
                - In nghiêng thông tin thời gian, địa điểm
                - Giữ nguyên định dạng của các số văn bản, công văn
                - Tạo bảng cho các dữ liệu số liệu

                5. Giữ nguyên:
                - Các số liệu và đơn vị
                - Mã số văn bản
                - Dấu câu và ký tự đặc biệt
                - Định dạng tiếng Việt

                Lưu ý: Đảm bảo tính chính xác và thẩm mỹ của văn bản khi chuyển sang Markdown.
                """

                # Get response from Gemini with retry
                try:
                    response = self._call_gemini_api(prompt, img_processed)
                    if response.text:
                        all_text.append(response.text)
                    print(f"✅ Đã xử lý trang {page_num + 1}/{len(pdf_document)}")
                except Exception as e:
                    print(f"❌ Lỗi API trang {page_num + 1}: {str(e)}")
                    raise

            # Join all text and check Vietnamese
            print("🔍 Đang kiểm tra và định dạng Markdown...")
            combined_text = "\n\n".join(all_text)
            corrected_text = self.check_vietnamese_text(combined_text)

            # Create output filename
            pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            os.makedirs(self.output_dir, exist_ok=True)

            # Save as Markdown
            output_path = os.path.join(self.output_dir, f"{pdf_filename}.md")
            original_path = os.path.join(self.output_dir, f"{pdf_filename}_original.md")

            # Clean and save both versions
            cleaned_combined = self.clean_markdown_content(combined_text)
            cleaned_corrected = self.clean_markdown_content(corrected_text)

            with open(original_path, "w", encoding="utf-8") as f:
                f.write(cleaned_combined)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_corrected)

            print(f"✅ Đã lưu văn bản gốc: {original_path}")
            print(f"✅ Đã lưu văn bản đã định dạng: {output_path}")

            pdf_document.close()
            return output_path

        except Exception as e:
            print(f"❌ Lỗi chuyển đổi PDF: {e}")
            return None

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
