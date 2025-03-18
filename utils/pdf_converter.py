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

import logging
from tqdm import tqdm


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
                logging.info(
                    f"✅ Đã kết nối Gemini API (Key {self.current_key_index + 1}/{len(self.api_keys)})"
                )

                # Initialize rate limiting tracking for each key
                # Gemini limits: 15 RPM, 1M TPM, 1,500 RPD
                self.key_usage = {}
                for i, _ in enumerate(self.api_keys):
                    self.key_usage[i] = {
                        "minute_requests": [],  # List of timestamps for RPM tracking
                        "day_requests": 0,  # Counter for RPD tracking
                        "day_reset": time.time() + 86400,  # Next day reset time
                        "tokens_used_minute": 0,  # Tokens used in current minute
                        "minute_reset": time.time() + 60,  # Next minute reset time
                    }

        except Exception as e:
            logging.error(f"❌ Lỗi cấu hình Gemini API: {e}")
            raise

        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "output"
        )
        self.retry_count = 0
        self.max_retries = 3
        self.wait_time = 60  # seconds

        # Rate limits for Gemini API
        self.RPM_LIMIT = 15  # Requests per minute
        self.TPM_LIMIT = 1000000  # Tokens per minute
        self.RPD_LIMIT = 1500  # Requests per day

        # Track rotation to prevent endless cycling
        self._rotation_cycle_count = 0
        self._last_rotation_time = time.time()

    def _update_rate_limits(self, estimated_tokens=0):
        """Update rate limit tracking for current key"""
        key_index = self.current_key_index
        current_time = time.time()

        # Update minute-based tracking
        self.key_usage[key_index]["minute_requests"].append(current_time)
        self.key_usage[key_index]["tokens_used_minute"] += estimated_tokens

        # Remove timestamps older than 1 minute
        self.key_usage[key_index]["minute_requests"] = [
            t
            for t in self.key_usage[key_index]["minute_requests"]
            if t > current_time - 60
        ]

        # Check if minute reset time passed
        if current_time > self.key_usage[key_index]["minute_reset"]:
            self.key_usage[key_index]["tokens_used_minute"] = estimated_tokens
            self.key_usage[key_index]["minute_reset"] = current_time + 60
            logging.info(f"🔄 Reset minute-based limits for API key {key_index + 1}")

        # Update day-based tracking
        self.key_usage[key_index]["day_requests"] += 1

        # Check if day reset time passed
        if current_time > self.key_usage[key_index]["day_reset"]:
            self.key_usage[key_index]["day_requests"] = 1
            self.key_usage[key_index]["day_reset"] = current_time + 86400
            logging.info(f"🔄 Reset daily limits for API key {key_index + 1}")

    def _check_rate_limits(self):
        """Check if current key is within rate limits"""
        key_index = self.current_key_index
        current_usage = self.key_usage[key_index]

        # Check RPM limit
        rpm_current = len(current_usage["minute_requests"])
        if rpm_current >= self.RPM_LIMIT:
            logging.warning(
                f"⚠️ API key {key_index + 1} đạt giới hạn RPM ({rpm_current}/{self.RPM_LIMIT})"
            )
            return False

        # Check TPM limit
        if current_usage["tokens_used_minute"] >= self.TPM_LIMIT:
            logging.warning(
                f"⚠️ API key {key_index + 1} đạt giới hạn TPM ({current_usage['tokens_used_minute']}/{self.TPM_LIMIT})"
            )
            return False

        # Check RPD limit
        if current_usage["day_requests"] >= self.RPD_LIMIT:
            logging.warning(
                f"⚠️ API key {key_index + 1} đạt giới hạn RPD ({current_usage['day_requests']}/{self.RPD_LIMIT})"
            )
            return False

        return True

    def _find_available_key(self):
        """Find an API key that hasn't reached its rate limits"""
        original_key = self.current_key_index

        # Try each key once
        for _ in range(len(self.api_keys)):
            # Check if current key is within limits
            if self._check_rate_limits():
                return True

            # Rotate to next key
            self.rotate_api_key()

            # If we've tried all keys and came back to the original one
            if self.current_key_index == original_key:
                break

        # If all keys are rate-limited
        wait_time = self._calculate_min_wait_time()
        logging.error(
            f"❌ All API keys are rate-limited. Wait at least {wait_time} seconds."
        )
        return False

    def _calculate_min_wait_time(self):
        """Calculate minimum time to wait for any key to become available again"""
        current_time = time.time()
        min_wait = 60  # Default 1 minute

        for key_index in self.key_usage:
            usage = self.key_usage[key_index]

            # Check minute-based reset
            if len(usage["minute_requests"]) >= self.RPM_LIMIT:
                # Find oldest request timestamp
                oldest = min(usage["minute_requests"])
                # Time until this falls outside the 1-minute window
                wait_time = (oldest + 60) - current_time
                min_wait = min(min_wait, max(1, wait_time))

            # If we're at TPM limit, we need to wait until minute reset
            if usage["tokens_used_minute"] >= self.TPM_LIMIT:
                wait_time = usage["minute_reset"] - current_time
                min_wait = min(min_wait, max(1, wait_time))

        return int(min_wait)

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _call_gemini_api(self, prompt, image=None):
        """Simplified API call with error handling and key rotation"""
        try:
            # Reset cycle detection if enough time has passed
            current_time = time.time()
            if (current_time - getattr(self, "_last_reset_time", 0)) > 60:
                if hasattr(self, "rotation_count"):
                    self.rotation_count = 0
                if hasattr(self, "_tried_all_keys"):
                    self._tried_all_keys = False
                if hasattr(self, "_exhausted_keys"):
                    self._exhausted_keys = set()
                self._last_reset_time = current_time

            # Check if we're cycling keys too rapidly
            if hasattr(self, "rotation_count") and self.rotation_count >= len(
                self.api_keys
            ):
                logging.warning(
                    "⚠️ Đã thử tất cả API key trong thời gian ngắn. Tạm dừng 10 giây..."
                )
                time.sleep(10)  # Take a shorter break before retrying
                self.rotation_count = 0

            # Check if current key is within rate limits, if not find one that is
            if not self._check_rate_limits():
                if not self._find_available_key():
                    wait_time = self._calculate_min_wait_time()
                    logging.warning(f"⏱️ Đang đợi {wait_time} giây cho API key reset...")
                    time.sleep(wait_time)

            # Estimate token usage for tracking
            prompt_tokens = self.estimate_tokens(text=prompt)
            image_tokens = 0
            if image:
                if isinstance(image, Image.Image):
                    image_tokens = self.estimate_tokens(image=image)
                else:
                    image_tokens = 1000  # Rough estimate for image bytes

            estimated_tokens = prompt_tokens + image_tokens

            # Make API call
            try:
                model = "gemini-2.0-flash"

                # Make API call with selected model
                if image:
                    # Check if the image is a PIL Image object and convert to bytes if needed
                    if isinstance(image, Image.Image):
                        # Convert PIL Image to bytes
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format="JPEG", quality=85)
                        img_bytes = img_bytes.getvalue()
                    else:
                        # Assume it's already bytes
                        img_bytes = image

                    # Create a temporary file for the image
                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                    ) as temp_img:
                        temp_img.write(img_bytes)
                        temp_img_path = temp_img.name

                    try:
                        # Upload the image file to Gemini
                        file_ref = self.client.files.upload(file=temp_img_path)

                        # Call Gemini with text prompt and file reference
                        response = self.client.models.generate_content(
                            model=model, contents=[prompt, file_ref]
                        )
                    finally:
                        # Clean up the temporary file
                        try:
                            os.unlink(temp_img_path)
                        except:
                            pass
                else:
                    # Text-only request
                    response = self.client.models.generate_content(
                        model=model, contents=[prompt]
                    )

                # Success - update rate limit tracking
                self._update_rate_limits(estimated_tokens)

                # Reset rotation counter and tried_all_keys flag
                if hasattr(self, "rotation_count"):
                    self.rotation_count = 0
                if hasattr(self, "_tried_all_keys"):
                    self._tried_all_keys = False
                if hasattr(self, "_exhausted_keys"):
                    self._exhausted_keys.clear()

                return response

            except Exception as e:
                error_message = str(e).lower()
                error_code = None

                # Extract error code if present
                if "429" in error_message:
                    error_code = 429
                elif "'code': " in error_message:
                    try:
                        code_str = (
                            error_message.split("'code': ")[1].split(",")[0].strip()
                        )
                        if code_str.isdigit():
                            error_code = int(code_str)
                    except:
                        pass

                # Check if this is a rate limit/resource exhausted error
                is_rate_limit_error = (
                    error_code == 429
                    or "resource exhausted" in error_message
                    or "rate limit" in error_message
                    or "too many requests" in error_message
                )

                # Check if this is a quota error (different from rate limit)
                is_quota_error = (
                    "quota exceeded" in error_message or "usage limit" in error_message
                )

                # Handle resource exhaustion by adding delay before retrying
                if is_rate_limit_error:
                    logging.warning(
                        f"⚠️ API key {self.current_key_index + 1}/{len(self.api_keys)} đang bị giới hạn tạm thời (rate limit)"
                    )

                    # Add a delay to recover from rate limiting
                    delay = 5 + (
                        self.retry_count * 5
                    )  # Increasing delay with each retry
                    logging.info(f"⏱️ Đợi {delay} giây cho rate limit reset...")
                    time.sleep(delay)

                    # Try with a different key if we have multiple keys
                    if len(self.api_keys) > 1:
                        # Try rotating to a new key
                        previous_key = self.current_key_index
                        self.rotate_api_key()
                        logging.info(
                            f"🔄 Đổi từ API key {previous_key + 1} sang {self.current_key_index + 1}/{len(self.api_keys)} do rate limit"
                        )
                        return self._call_gemini_api(prompt, image)

                    # For single key, just raise to let tenacity retry
                    raise

                # Handle permanent quota exhaustion
                elif is_quota_error:
                    # Update tracking for this key to mark it as exhausted for today
                    self.key_usage[self.current_key_index]["day_requests"] = (
                        self.RPD_LIMIT
                    )

                    logging.warning(
                        f"⚠️ API key {self.current_key_index + 1}/{len(self.api_keys)} đã hết quota."
                    )

                    # Initialize exhausted keys tracking if needed
                    if not hasattr(self, "_exhausted_keys"):
                        self._exhausted_keys = set()

                    # Mark current key as tried
                    self._exhausted_keys.add(self.current_key_index)

                    # If we have multiple keys, try to find an unused one
                    if len(self.api_keys) > 1:
                        # Try each key at most once
                        for _ in range(len(self.api_keys) - 1):
                            self.rotate_api_key()

                            # Skip this key if we've already tried it
                            if self.current_key_index in self._exhausted_keys:
                                continue

                            logging.info(
                                f"🔄 Đã chuyển sang API key {self.current_key_index + 1}/{len(self.api_keys)} do hết quota"
                            )
                            return self._call_gemini_api(prompt, image)

                        # If we've tried all keys during this attempt
                        if len(self._exhausted_keys) >= len(self.api_keys):
                            logging.error(
                                f"❌ Đã thử tất cả {len(self.api_keys)} API keys. Có thể tất cả đều hết quota."
                            )
                            raise Exception("ALL_KEYS_EXHAUSTED")
                    else:
                        # Only one key and it's exhausted
                        logging.error("❌ API key duy nhất đã hết quota.")
                        raise Exception("API_QUOTA_EXHAUSTED")
                else:
                    # Handle other errors - log the full error for debugging
                    logging.warning(f"⚠️ API error (non-quota): {str(e)}")

                    # Add a small delay for other errors
                    time.sleep(1)
                    raise

        except Exception as e:
            if "ALL_KEYS_EXHAUSTED" in str(e) or "API_QUOTA_EXHAUSTED" in str(e):
                # Forward these specific errors to calling function
                raise

            logging.error(f"❌ API error: {str(e)}")

            # If we've retried too many times, take a break
            if self.retry_count >= self.max_retries:
                logging.error(
                    f"⚠️ Maximum retries reached. Waiting {self.wait_time} seconds..."
                )
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
        logging.info(f"✅ Đã điều chỉnh quota tracker cho {len(new_keys)} API keys")

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
            logging.warning(
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
                        logging.info(
                            f"🔄 Đã chuyển sang API key {self.current_key_index + 1} vì key {original_key + 1} đang bị giới hạn"
                        )
                        return True

                # If we're here, all keys are exhausted
                logging.error(wait_message)
                return False
            else:
                # We only have one key and it's exhausted
                logging.error(wait_message)
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

            # Track rotation time
            self._last_rotation_time = time.time()

            # Track rotations
            if not hasattr(self, "rotation_count"):
                self.rotation_count = 0
            self.rotation_count += 1

            logging.info(
                f"🔄 Đã chuyển sang API key {self.current_key_index + 1}/{len(self.api_keys)}"
            )

            return True
        except Exception as e:
            logging.error(f"❌ Lỗi khi chuyển API key: {str(e)}")
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

            logging.info("✨ Đã xử lý ảnh để tăng chất lượng OCR")
            return img_cropped

        except Exception as e:
            logging.warning(f"⚠️ Lỗi xử lý ảnh: {e}")
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
            logging.warning(f"⚠️ Lỗi kiểm tra chính tả: {e}")
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
            logging.warning(f"⚠️ Lỗi khi làm sạch Markdown: {e}")
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
        start_time = time.time()

        # Define progress file path based on PDF name
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        progress_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "progress"
        )
        os.makedirs(progress_dir, exist_ok=True)
        progress_file = os.path.join(progress_dir, f"{pdf_filename}_progress.json")

        # Initialize or load progress data
        processed_pages = {}
        current_page_retries = {}
        start_page = 0

        if resume and os.path.exists(progress_file):
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    progress_data = json.load(f)
                    processed_pages = progress_data.get("pages", {})
                    estimated_tokens_used = progress_data.get("tokens_used", 0)
                    current_page_retries = progress_data.get("page_retries", {})
                    logging.info(
                        f"🔄 Tiếp tục xử lý từ tiến độ đã lưu. Đã xử lý {len(processed_pages)} trang."
                    )
            except Exception as e:
                logging.warning(f"⚠️ Không thể tải tiến độ đã lưu: {str(e)}")

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)

            if pdf_document.page_count == 0:
                logging.warning(
                    f"⚠️ PDF không có trang nào: {os.path.basename(pdf_path)}"
                )
                return False

            all_text = []
            logging.info(f"🔍 Đang chuyển đổi {pdf_path}...")

            # Tracking variables
            success_count = len(processed_pages)
            failed_count = 0
            total_pages = pdf_document.page_count

            # Hiển thị thông tin PDF
            logging.info(
                f"📄 File PDF có {total_pages} trang, đã xử lý {success_count} trang"
            )

            # Khởi tạo thanh tiến độ với tqdm
            pbar = tqdm(
                total=total_pages,
                initial=success_count,
                desc="Xử lý trang PDF",
                unit="trang",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
            try:
                for page_num in range(total_pages):
                    # Skip already processed pages if resuming
                    page_key = str(page_num)
                    if resume and page_key in processed_pages:
                        all_text.append(processed_pages[page_key])
                        continue

                    retry_count = current_page_retries.get(page_key, 0)
                    max_page_retries = 2
                    page_processed = False

                    while retry_count <= max_page_retries and not page_processed:
                        try:
                            # Get the page with error handling
                            try:
                                page = pdf_document[page_num]
                            except IndexError:
                                logging.warning(
                                    f"⚠️ Lỗi khi truy cập trang {page_num + 1}"
                                )
                                failed_count += 1
                                break

                            if not page:
                                logging.warning(f"⚠️ Không thể đọc trang {page_num + 1}")
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
                                logging.warning(
                                    f"⚠️ Lỗi khi tạo ảnh trang {page_num + 1}: {str(img_err)}"
                                )
                                retry_count += 1
                                continue

                            # Preprocess image
                            img_processed = self.preprocess_image(img)

                            # Sử dụng prompt ngắn hơn để tiết kiệm token
                            prompt = """
                            Chuyển thành Markdown:
                            - # tiêu đề chính, ## tiêu đề cấp 2, ### phần chính
                            - **CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM**
                            - *Thời gian, địa điểm*
                            - > trích dẫn, --- đường kẻ
                            - | và - cho bảng (số căn phải)
                            - - danh sách không thứ tự, 1. danh sách thứ tự
                            Giữ nguyên mã văn bản, số liệu, dấu câu, định dạng tiếng Việt.
                            """

                            prompt_tokens = self.estimate_tokens(prompt)
                            image_tokens = self.estimate_tokens(image=img_processed)
                            request_tokens = prompt_tokens + image_tokens

                            # Check if we're over budget
                            if token_budget and (
                                estimated_tokens_used + request_tokens > token_budget
                            ):
                                logging.warning(
                                    f"⚠️ Đã vượt ngân sách token ({token_budget}). Dừng xử lý."
                                )
                                break

                            try:
                                response = self._call_gemini_api(prompt, img_processed)

                                # Cập nhật số tokens đã sử dụng
                                estimated_tokens_used += request_tokens

                                if (
                                    response
                                    and hasattr(response, "text")
                                    and response.text
                                ):
                                    all_text.append(response.text)
                                    processed_pages[page_key] = response.text
                                    success_count += 1
                                    page_processed = True

                                    # Cập nhật thanh tiến độ
                                    pbar.update(1)
                                    # Thêm thông tin token vào thanh tiến độ
                                    pbar.set_postfix(tokens=estimated_tokens_used)

                                    # Lưu tiến độ sau mỗi trang thành công
                                    self._save_progress_with_retries(
                                        progress_file,
                                        processed_pages,
                                        estimated_tokens_used,
                                        current_page_retries,
                                    )
                                else:
                                    logging.warning(
                                        f"⚠️ Không có phản hồi từ API cho trang {page_num + 1}"
                                    )
                                    retry_count += 1
                                    current_page_retries[page_key] = retry_count
                            except Exception as api_err:
                                error_msg = str(api_err)
                                if (
                                    "API_QUOTA_EXHAUSTED" in error_msg
                                    or "ALL_KEYS_EXHAUSTED" in error_msg
                                ):
                                    logging.warning(
                                        f"⚠️ Tất cả API key đã hết quota khi xử lý trang {page_num + 1}"
                                    )
                                    # Lưu tiến độ để tiếp tục sau
                                    self._save_progress_with_retries(
                                        progress_file,
                                        processed_pages,
                                        estimated_tokens_used,
                                        current_page_retries,
                                    )
                                    pbar.close()
                                    logging.info(
                                        "💾 Đã lưu tiến độ. Có thể tiếp tục xử lý sau với 'resume=True'"
                                    )
                                    return False
                                else:
                                    logging.error(
                                        f"❌ Lỗi API trang {page_num + 1}: {str(api_err)}"
                                    )
                                    retry_count += 1
                                    current_page_retries[page_key] = retry_count
                                    if retry_count <= max_page_retries:
                                        # Thử đổi key nếu có lỗi
                                        self.rotate_api_key()
                                        time.sleep(1)  # Tạm dừng ngắn trước khi thử lại

                        except Exception as page_err:
                            logging.warning(
                                f"⚠️ Lỗi xử lý trang {page_num + 1}: {str(page_err)}"
                            )
                            retry_count += 1
                            current_page_retries[page_key] = retry_count

                    # Count as failed if all retries were used up without success
                    if not page_processed:
                        failed_count += 1

            except KeyboardInterrupt:
                # Đóng thanh tiến độ nếu đang hiển thị
                if "pbar" in locals():
                    pbar.close()

                # Lưu tiến độ trước khi thoát
                logging.warning(
                    "⚠️ Phát hiện lệnh dừng từ người dùng (Ctrl+C). Đang lưu tiến độ..."
                )
                self._save_progress_with_retries(
                    progress_file,
                    processed_pages,
                    estimated_tokens_used,
                    current_page_retries,
                )
                logging.info(
                    f"💾 Đã lưu tiến độ ({len(processed_pages)}/{total_pages} trang). Có thể tiếp tục với resume=True"
                )

                # Đóng PDF document trước khi thoát
                if pdf_document:
                    try:
                        pdf_document.close()
                    except Exception:
                        pass

                return False  # Return False to indicate incomplete processing

            # Đóng thanh tiến độ khi hoàn tất
            pbar.close()

            # Check if we processed any pages successfully
            if not all_text:
                logging.error(
                    f"❌ Không thể xử lý bất kỳ trang nào của PDF ({failed_count}/{total_pages} lỗi)"
                )
                return False

            total_time = time.time() - start_time
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            logging.info(
                f"📊 Kết quả xử lý: {success_count}/{total_pages} trang thành công "
                f"({success_count / total_pages * 100:.1f}%) trong {minutes}m {seconds}s"
            )

            # Join all text and process
            try:
                logging.info("🔍 Đang kiểm tra và định dạng Markdown...")
                combined_text = "\n\n".join(all_text)

                try:
                    corrected_text = self.check_vietnamese_text(combined_text)
                except Exception as vn_err:
                    logging.warning(f"⚠️ Lỗi kiểm tra tiếng Việt: {str(vn_err)}")
                    corrected_text = combined_text

                # Create output filename
                pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
                os.makedirs(self.output_dir, exist_ok=True)
                output_path = os.path.join(self.output_dir, f"{pdf_filename}.md")

                try:
                    cleaned_text = self.clean_markdown_content(corrected_text)
                except Exception as clean_err:
                    logging.warning(f"⚠️ Lỗi làm sạch Markdown: {str(clean_err)}")
                    cleaned_text = corrected_text

                # Save the final text
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned_text)

                logging.info(f"✅ Đã lưu văn bản: {output_path}")
                return output_path

            except Exception as process_err:
                logging.error(f"❌ Lỗi xử lý văn bản cuối cùng: {str(process_err)}")

                # Try to save raw text if final processing fails
                try:
                    emergency_text = "\n\n".join(all_text)
                    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
                    emergency_path = os.path.join(
                        self.output_dir, f"{pdf_filename}_emergency.md"
                    )

                    with open(emergency_path, "w", encoding="utf-8") as f:
                        f.write(emergency_text)

                    logging.error(f"⚠️ Đã lưu văn bản khẩn cấp: {emergency_path}")
                    return emergency_path
                except Exception as emergency_err:
                    logging.error(
                        f"❌ Không thể lưu văn bản khẩn cấp: {str(emergency_err)}"
                    )
                    return False

        except Exception as e:
            logging.error(f"❌ Lỗi chuyển đổi PDF: {str(e)}")
            # Save progress on exception too
            self._save_progress_with_retries(
                progress_file,
                processed_pages,
                estimated_tokens_used,
                current_page_retries,
            )
            if "pbar" in locals():
                pbar.close()
            return False

        finally:
            # Make sure to close the PDF document to release the file handle
            if pdf_document:
                try:
                    pdf_document.close()
                except Exception as close_err:
                    logging.warning(f"⚠️ Lỗi khi đóng file PDF: {str(close_err)}")

    def cleanup(self):
        """Clean up temporary resources after processing"""
        try:
            # Clean up temp directory if it exists
            if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
                try:
                    import shutil

                    shutil.rmtree(self.temp_dir)
                    logging.info(f"✅ Đã dọn dẹp thư mục tạm: {self.temp_dir}")
                except Exception as e:
                    logging.warning(f"⚠️ Không thể dọn dẹp thư mục tạm: {str(e)}")

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
                logging.warning(f"⚠️ Lỗi lưu trữ quota tracker: {str(quota_err)}")

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
                                logging.warning(f"🧹 Đã xóa tiến độ cũ: {filename}")
                except Exception as e:
                    logging.warning(f"⚠️ Không thể dọn dẹp file tiến độ: {str(e)}")

            logging.info("✅ Đã dọn dẹp tài nguyên tạm thời")
        except Exception as e:
            logging.warning(f"⚠️ Lỗi khi dọn dẹp: {str(e)}")
