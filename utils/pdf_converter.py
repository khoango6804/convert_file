import io
import os
import fitz
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from google import genai
import tempfile
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime, timedelta


def estimate_tokens(self, text=None, image=None):
    """Estimate token usage of a request"""
    token_count = 0

    # Text token estimation (approx 4 chars per token)
    if text:
        token_count += len(text) / 4

    # Image token estimation based on dimensions
    if image and hasattr(image, "width") and hasattr(image, "height"):
        # Gemini charges more for larger images
        pixels = image.width * image.height
        token_count += min(pixels / 750, 4000)  # Rough estimate

    return int(token_count)


class PDFConverter:
    def __init__(self):
        # Load API key from config
        try:
            with open(
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "api.json"),
                "r",
            ) as f:
                config = json.load(f)

                # Support both single key and multiple keys formats
                if "api_keys" in config:
                    self.api_keys = config["api_keys"]
                    self.single_key = False
                elif "api_key" in config:
                    self.api_keys = [config["api_key"]]
                    self.single_key = True
                else:
                    raise ValueError("No API keys found in config")

                # Use the first key initially
                self.current_key_index = 0

                # Configure Gemini
                self.client = genai.Client(
                    api_key=self.api_keys[self.current_key_index]
                )
                print(
                    f"✅ Đã kết nối Gemini API (Key {self.current_key_index + 1}/{len(self.api_keys)})"
                )

        except Exception as e:
            print(f"❌ Lỗi cấu hình Gemini API: {e}")
            raise

        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "output"
        )
        self.retry_count = 0
        self.max_retries = 3
        self.wait_time = 60  # seconds

        # Track rotation to prevent endless cycling
        self._rotation_cycle_count = 0
        self._last_rotation_time = time.time()

    def _save_progress(self, progress_file, processed_pages, tokens_used):
        """Save processing progress to allow resuming later"""
        try:
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "pages": processed_pages,
                "tokens_used": tokens_used,
            }

            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)

            print(f"💾 Đã lưu tiến độ ({len(processed_pages)} trang)")
        except Exception as e:
            print(f"⚠️ Không thể lưu tiến độ: {str(e)}")

    def _prepare_image_for_api(self, image):
        """Prepare image for API with quota optimization"""
        try:
            # Ensure we have a valid image
            if not isinstance(image, Image.Image):
                print("⚠️ Invalid image provided")
                return None

            # Resize image more aggressively for token savings
            width, height = image.size
            max_dimension = 1024  # Reduced from 1600 to save tokens
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))
                image = image.resize((new_width, new_height), Image.LANCZOS)
                print(
                    f"🔍 Đã resize hình ảnh từ {width}x{height} thành {new_width}x{new_height}"
                )

            # Convert to grayscale to reduce tokens
            if image.mode != "L":
                image = ImageOps.grayscale(image)
                print("🔍 Đã chuyển ảnh sang grayscale để giảm token")

            # Compress image quality
            img_bytes = io.BytesIO()
            image.save(
                img_bytes, format="JPEG", quality=85
            )  # Using JPEG with 85% quality
            img_bytes.seek(0)

            return img_bytes.getvalue()

        except Exception as e:
            print(f"⚠️ Error preparing image: {str(e)}")
            return None

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_gemini_api(self, prompt, image=None):
        """Simplified API call with error handling and key rotation"""
        try:
            # Check if we're cycling keys too rapidly
            if (
                hasattr(self, "rotation_count")
                and self.rotation_count >= len(self.api_keys) * 2
            ):
                print("⚠️ Đang quay vòng API key quá nhanh. Tạm dừng...")
                time.sleep(30)  # Take a longer break
                self.rotation_count = 0

            # Make API call
            try:
                # Choose most efficient model based on content
                if image:
                    # Use flash for images to save tokens
                    model = "gemini-2.0-flash"
                else:
                    # For text-only, use Pro which is more efficient with text
                    model = "gemini-2.0-pro"

                # Make API call with selected model
                if image:
                    response = self.client.models.generate_content(
                        model=model, contents=[prompt, image]
                    )
                else:
                    response = self.client.models.generate_content(
                        model=model, contents=[prompt]
                    )

                # Success! Reset rotation counter
                if hasattr(self, "rotation_count"):
                    self.rotation_count = 0

                return response

            except Exception as e:
                error_message = str(e).lower()

                # Check if all keys are exhausted
                if any(msg in error_message for msg in ["quota", "rate", "limit"]):
                    if len(self.api_keys) > 1:
                        # Try another key as you're already doing
                        self.rotate_api_key()
                        return self._call_gemini_api(prompt, image)
                    else:
                        # Single key case - save progress before waiting
                        print("⚠️ Hết hạn API, đã lưu tiến độ. Có thể tiếp tục sau.")
                        # Let the error propagate so the main function can save progress
                        raise Exception("API_QUOTA_EXHAUSTED")

                # Re-raise for other types of errors
                raise

        except Exception as e:
            print(f"❌ API error: {str(e)}")

            # If we've retried too many times, take a break
            if self.retry_count >= self.max_retries:
                print(f"⚠️ Maximum retries reached. Waiting {self.wait_time} seconds...")
                time.sleep(self.wait_time)
                self.retry_count = 0
            else:
                self.retry_count += 1

            raise

    def _adjust_quota_tracker(self):
        """Adjust quota tracker to match current API keys"""
        current_keys = self.quota_tracker["keys"].copy()
        new_keys = []

        # Keep existing data for keys we already have
        for i in range(len(self.api_keys)):
            if i < len(current_keys):
                new_keys.append(current_keys[i])
            else:
                # Add new entries for additional keys
                tomorrow = (
                    datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    + timedelta(days=1)
                ).isoformat()

                new_keys.append({"daily_requests": 0, "next_allowed_time": tomorrow})

        # Update the tracker
        self.quota_tracker["keys"] = new_keys
        print(f"✅ Đã điều chỉnh quota tracker cho {len(new_keys)} API keys")

    def save_quota_tracker(self):
        """Save quota tracker to file"""
        with open(self.quota_tracker_file, "w") as f:
            json.dump(self.quota_tracker, f, indent=2)

    def update_key_usage(self):
        """Update usage for current key and check if we need to wait"""
        key_data = self.quota_tracker["keys"][self.current_key_index]
        key_data["daily_requests"] += 1

        # If we've reached the daily limit (1500 for free tier)
        if key_data["daily_requests"] >= 1500:
            # Set next allowed time to 24 hours from now
            next_time = datetime.now() + timedelta(hours=24)
            key_data["next_allowed_time"] = next_time.isoformat()
            print(
                f"⚠️ Daily limit reached for key {self.current_key_index + 1}. Next allowed: {next_time}"
            )

        self.save_quota_tracker()
        return key_data["daily_requests"] >= 1500

    def check_key_availability(self):
        """Check if current key is available or if we need to wait/rotate"""
        key_data = self.quota_tracker["keys"][self.current_key_index]
        next_allowed_time = datetime.fromisoformat(key_data["next_allowed_time"])

        if datetime.now() < next_allowed_time:
            # Key is not available yet
            time_diff = next_allowed_time - datetime.now()
            hours, remainder = divmod(time_diff.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            wait_message = f"⚠️ API key {self.current_key_index + 1} đang bị giới hạn. "
            wait_message += f"Tiếp tục sau: {hours} giờ, {minutes} phút, {seconds} giây"

            if len(self.api_keys) > 1:
                # Try other keys
                original_key = self.current_key_index
                for _ in range(len(self.api_keys) - 1):
                    self.rotate_api_key()
                    if self.check_key_availability():
                        print(
                            f"🔄 Đã chuyển sang API key {self.current_key_index + 1} vì key {original_key + 1} đang bị giới hạn"
                        )
                        return True

                # If we're here, all keys are exhausted
                print(wait_message)
                return False
            else:
                # We only have one key and it's exhausted
                print(wait_message)
                return False

        return True  # Key is available

    def rotate_api_key(self):
        """Simple API key rotation without quota tracking"""
        if len(self.api_keys) <= 1:
            return False  # Can't rotate with only one key

        # Store previous key index for logging
        previous_key = self.current_key_index

        # Move to next key
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

        # Configure client with new key
        try:
            self.client = genai.Client(api_key=self.api_keys[self.current_key_index])

            # Only log if we're not cycling too rapidly
            current_time = time.time()
            if current_time - self._last_rotation_time > 5:
                print(
                    f"🔄 Đã chuyển sang API key {self.current_key_index + 1}/{len(self.api_keys)}"
                )

            self._last_rotation_time = current_time

            # Track rotations to detect cycling
            if not hasattr(self, "rotation_count"):
                self.rotation_count = 0
            self.rotation_count += 1

            # If we've rotated through all keys multiple times in a short period, take a break
            if (
                self.rotation_count >= len(self.api_keys) * 2
                and current_time - self._last_rotation_time < 30
            ):
                print("⚠️ Đã thử tất cả API keys nhiều lần. Tạm dừng 30 giây...")
                time.sleep(30)
                self.rotation_count = 0  # Reset counter

            return True
        except Exception as e:
            print(f"❌ Lỗi khi chuyển API key: {str(e)}")
            self.current_key_index = previous_key  # Revert to previous key
            return False

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
            Chuẩn hóa văn bản hành chính thành Markdown:

            1. Tiêu đề:
            - # cho cơ quan cao nhất 
            - ## cho cơ quan trực thuộc
            - ### cho số văn bản/trích yếu

            2. Định dạng:
            - **CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM**
            - *Độc lập - Tự do - Hạnh phúc*
            - Xuống dòng: 2 dấu cách cuối dòng
            - Căn phải: 4 dấu cách đầu dòng

            3. Danh sách:
            - Có dòng trống trước/sau
            - - danh sách không thứ tự
            - 1. 2. danh sách có thứ tự
            - > cho trích dẫn

            4. Bảng và ghi chú:
            - | và - cho bảng, :--- căn trái, ---: căn phải
            - Ghi chú dạng [^n] thay cho <sup>n</sup>

            Không dùng HTML, không để dấu câu cuối tiêu đề, thêm dòng trống cuối file.
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

    def estimate_tokens(self, text=None, image=None):
        """Estimate token usage of a request"""
        token_count = 0

        # Text token estimation (approx 4 chars per token)
        if text:
            token_count += len(text) / 4

        # Image token estimation based on dimensions
        if image and hasattr(image, "width") and hasattr(image, "height"):
            # Gemini charges more for larger images
            pixels = image.width * image.height
            token_count += min(pixels / 750, 4000)  # Rough estimate

        return int(token_count)

    def pdf_to_text(
        self, pdf_path, output_format="md", token_budget=None, resume=False
    ):
        """
        Convert PDF with optional token budget
        Args:
            token_budget: Maximum tokens to use (None for unlimited)
        """
        estimated_tokens_used = 0
        pdf_document = None

        # Define progress file path based on PDF name
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        progress_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "progress"
        )
        os.makedirs(progress_dir, exist_ok=True)
        progress_file = os.path.join(progress_dir, f"{pdf_filename}_progress.json")

        # Initialize or load progress data
        processed_pages = {}
        start_page = 0

        if resume and os.path.exists(progress_file):
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    progress_data = json.load(f)
                    processed_pages = progress_data.get("pages", {})
                    estimated_tokens_used = progress_data.get("tokens_used", 0)
                    print(
                        f"🔄 Tiếp tục xử lý từ tiến độ đã lưu. Đã xử lý {len(processed_pages)} trang."
                    )
            except Exception as e:
                print(f"⚠️ Không thể tải tiến độ đã lưu: {str(e)}")

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)

            if pdf_document.page_count == 0:
                print(f"⚠️ PDF không có trang nào: {os.path.basename(pdf_path)}")
                return False

            all_text = []
            print(f"🔍 Đang chuyển đổi {pdf_path}...")

            # Tracking variables
            success_count = 0
            failed_count = 0
            total_pages = pdf_document.page_count

            for page_num in range(total_pages):
                # Skip already processed pages if resuming
                page_key = str(page_num)
                if resume and page_key in processed_pages:
                    all_text.append(processed_pages[page_key])
                    print(f"⏩ Bỏ qua trang {page_num + 1} (đã xử lý trước đó)")
                    continue

                retry_count = 0
                max_page_retries = 2
                page_processed = False

                while retry_count <= max_page_retries and not page_processed:
                    try:
                        # Get the page with error handling
                        try:
                            page = pdf_document[page_num]
                        except IndexError:
                            print(f"⚠️ Lỗi khi truy cập trang {page_num + 1}")
                            failed_count += 1
                            break

                        if not page:
                            print(f"⚠️ Không thể đọc trang {page_num + 1}")
                            failed_count += 1
                            break

                        # Convert page to image
                        try:
                            pix = page.get_pixmap(
                                matrix=fitz.Matrix(200 / 72, 200 / 72)
                            )
                            img = Image.frombytes(
                                "RGB", [pix.width, pix.height], pix.samples
                            )
                        except Exception as img_err:
                            print(
                                f"⚠️ Lỗi khi tạo ảnh trang {page_num + 1}: {str(img_err)}"
                            )
                            retry_count += 1
                            continue

                        # Preprocess image
                        img_processed = self.preprocess_image(img)

                        # Create prompt for Gemini
                        prompt = """
                        Chuyển thành Markdown:

                        1. Tiêu đề:
                        - # tiêu đề chính
                        - ## tiêu đề cấp 2
                        - ### phần chính

                        2. Định dạng:
                        - **CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM**
                        - *Thời gian, địa điểm*
                        - > trích dẫn
                        - --- đường kẻ

                        3. Bảng/Danh sách:
                        - | và - cho bảng (số căn phải)
                        - - cho danh sách không thứ tự
                        - 1. cho danh sách thứ tự

                        Giữ nguyên: mã văn bản, số liệu, dấu câu, định dạng tiếng Việt.
                        """
                        prompt_tokens = self.estimate_tokens(prompt)
                        image_tokens = self.estimate_tokens(image=img_processed)
                        request_tokens = prompt_tokens + image_tokens

                        # Check if we're over budget
                        if token_budget and (
                            estimated_tokens_used + request_tokens > token_budget
                        ):
                            print(
                                f"⚠️ Token budget ({token_budget}) would be exceeded. Stopping."
                            )
                            break

                        # After successful API call:
                        estimated_tokens_used += request_tokens
                        print(
                            f"📊 Estimated tokens used so far: {estimated_tokens_used}"
                        )
                        # Try to call API with error handling
                        try:
                            response = self._call_gemini_api(prompt, img_processed)

                            if response and hasattr(response, "text") and response.text:
                                all_text.append(response.text)
                                success_count += 1
                                page_processed = True
                                print(f"✅ Đã xử lý trang {page_num + 1}/{total_pages}")
                            else:
                                print(
                                    f"⚠️ Không có phản hồi từ API cho trang {page_num + 1}"
                                )
                                retry_count += 1
                        except Exception as api_err:
                            print(f"❌ Lỗi API trang {page_num + 1}: {str(api_err)}")
                            retry_count += 1
                            # If we still have retries left, try with a different API key
                            if retry_count <= max_page_retries:
                                self.rotate_api_key()
                                time.sleep(2)  # Short pause before retrying

                    except Exception as page_err:
                        print(f"⚠️ Lỗi xử lý trang {page_num + 1}: {str(page_err)}")
                        retry_count += 1

                # Count as failed if all retries were used up without success
                if not page_processed:
                    failed_count += 1
                else:
                    processed_pages[page_key] = response.text
                    # Save progress after each successful page
                    self._save_progress(
                        progress_file, processed_pages, estimated_tokens_used
                    )

            # Check if we processed any pages successfully
            if not all_text:
                print(
                    f"❌ Không thể xử lý bất kỳ trang nào của PDF ({failed_count}/{total_pages} lỗi)"
                )
                return False

            print(
                f"📊 Kết quả xử lý: {success_count}/{total_pages} trang thành công ({success_count / total_pages * 100:.1f}%)"
            )

            # Join all text and process
            try:
                print("🔍 Đang kiểm tra và định dạng Markdown...")
                combined_text = "\n\n".join(all_text)

                try:
                    corrected_text = self.check_vietnamese_text(combined_text)
                except Exception as vn_err:
                    print(f"⚠️ Lỗi kiểm tra tiếng Việt: {str(vn_err)}")
                    corrected_text = combined_text

                # Create output filename
                pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
                os.makedirs(self.output_dir, exist_ok=True)
                output_path = os.path.join(self.output_dir, f"{pdf_filename}.md")

                try:
                    cleaned_text = self.clean_markdown_content(corrected_text)
                except Exception as clean_err:
                    print(f"⚠️ Lỗi làm sạch Markdown: {str(clean_err)}")
                    cleaned_text = corrected_text

                # Save the final text
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)

                print(f"✅ Đã lưu văn bản: {output_path}")
                return output_path

            except Exception as process_err:
                print(f"❌ Lỗi xử lý văn bản cuối cùng: {str(process_err)}")

                # Try to save raw text if final processing fails
                try:
                    emergency_text = "\n\n".join(all_text)
                    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
                    emergency_path = os.path.join(
                        self.output_dir, f"{pdf_filename}_emergency.md"
                    )

                    with open(emergency_path, "w", encoding="utf-8") as f:
                        f.write(emergency_text)

                    print(f"⚠️ Đã lưu văn bản khẩn cấp: {emergency_path}")
                    return emergency_path
                except Exception as emergency_err:
                    print(f"❌ Không thể lưu văn bản khẩn cấp: {str(emergency_err)}")
                    return False

        except Exception as e:
            print(f"❌ Lỗi chuyển đổi PDF: {str(e)}")
            # Save progress on exception too
            self._save_progress(progress_file, processed_pages, estimated_tokens_used)
            return False

        finally:
            # Make sure to close the PDF document to release the file handle
            if pdf_document:
                try:
                    pdf_document.close()
                except Exception as close_err:
                    print(f"⚠️ Lỗi khi đóng file PDF: {str(close_err)}")

    def cleanup(self):
        """Clean up temporary resources after processing"""
        try:
            # Clean up temp directory if it exists
            if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
                try:
                    import shutil

                    shutil.rmtree(self.temp_dir)
                    print(f"✅ Đã dọn dẹp thư mục tạm: {self.temp_dir}")
                except Exception as e:
                    print(f"⚠️ Không thể dọn dẹp thư mục tạm: {str(e)}")

            # Reset API rotation counters to prevent issues on next run
            if hasattr(self, "_rotation_cycle_count"):
                self._rotation_cycle_count = 0

            if hasattr(self, "_current_call_rotations"):
                self._current_call_rotations = 0

            if hasattr(self, "rotation_count"):
                self.rotation_count = 0

            if hasattr(self, "retry_count"):
                self.retry_count = 0

            # Save current state of quota tracker
            try:
                if hasattr(self, "quota_tracker") and hasattr(
                    self, "save_quota_tracker"
                ):
                    self.save_quota_tracker()
            except Exception as quota_err:
                print(f"⚠️ Lỗi lưu trữ quota tracker: {str(quota_err)}")

            # Clean up progress files for completed documents
            progress_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data", "progress"
            )
            if os.path.exists(progress_dir):
                # Only remove progress files older than 7 days
                cutoff = datetime.now() - timedelta(days=7)
                try:
                    for filename in os.listdir(progress_dir):
                        if filename.endswith("_progress.json"):
                            filepath = os.path.join(progress_dir, filename)
                            file_time = datetime.fromtimestamp(
                                os.path.getmtime(filepath)
                            )
                            if file_time < cutoff:
                                os.remove(filepath)
                                print(f"🧹 Đã xóa tiến độ cũ: {filename}")
                except Exception as e:
                    print(f"⚠️ Không thể dọn dẹp file tiến độ: {str(e)}")

            print("✅ Đã dọn dẹp tài nguyên tạm thời")
        except Exception as e:
            print(f"⚠️ Lỗi khi dọn dẹp: {str(e)}")
