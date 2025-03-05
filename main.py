import time
import os
import sys
import json
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from utils.log_config import setup_logging

from utils.doc_converter import extract_text_from_doc, extract_text_from_docx
from utils.file_utils import (
    ensure_folder_exists,
    move_to_error_folder,
    move_to_processed_folder,
    is_already_processed,
)
from utils.pdf_converter import PDFConverter
import logging


def setup_folders() -> Tuple[str, str, str, str]:  # Updated return type
    """Setup required folders and return their paths"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders = {
        "input": os.path.join(base_dir, "data", "input"),
        "output": os.path.join(base_dir, "data", "output"),
        "error": os.path.join(base_dir, "data", "error"),
        "processed": os.path.join(base_dir, "data", "processed"),
    }

    # Create folders if they don't exist
    for folder in folders.values():
        ensure_folder_exists(folder)

    return folders["input"], folders["output"], folders["error"], folders["processed"]


def check_input_files(
    input_folder: str, error_folder: str, output_folder: str, processed_folder: str
) -> Tuple[List[str], bool]:
    """Check input and error folders for valid files"""
    input_files = []
    error_files = []
    already_processed = []

    # Check input folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".doc", ".docx", ".pdf")):
                file_path = os.path.join(root, file)

                # Check if file has already been processed
                if is_already_processed(file_path, output_folder):
                    # Move directly to processed folder
                    logging.info(f"⏭️ Đã tìm thấy file đã xử lý: {file}")
                    move_to_processed_folder(file_path, processed_folder)
                    already_processed.append(file)
                else:
                    # New file, add to processing queue
                    input_files.append(file_path)

    # Check error folder
    for root, _, files in os.walk(error_folder):
        for file in files:
            if file.lower() != "error_log.txt" and file.lower().endswith(
                (".doc", ".docx", ".pdf")
            ):
                error_files.append(os.path.join(root, file))

    # Print summary of skipped files
    if already_processed:
        logging.info(
            f"⏭️ Đã chuyển {len(already_processed)} file đã xử lý trước đó sang thư mục processed"
        )

    # If input is empty but error has files, process error folder
    if not input_files and error_files:
        logging.warning(
            f"⚠️ Thư mục input trống. Tìm thấy {len(error_files)} file trong thư mục error."
        )
        logging.info("🔄 Chuyển sang xử lý files từ thư mục error...")
        return error_files, True

    # If input has files, process those
    if input_files:
        return input_files, False

    # If both are empty
    logging.warning("⚠️ Không tìm thấy file .doc, .docx hoặc .pdf nào để xử lý")
    return [], False


def process_file(
    input_path, output_folder, error_folder, processed_folder, pdf_converter, log_file
):
    """Process a single file and handle errors"""
    file = os.path.basename(input_path)
    file_lower = file.lower()
    processed_file_paths = set()
    file_types = {"pdf": 0, "doc": 0, "docx": 0}
    processed_files = []
    error_files = []

    # Count file types
    if file_lower.endswith(".pdf"):
        file_types["pdf"] += 1
    elif file_lower.endswith(".docx"):
        file_types["docx"] += 1
    elif file_lower.endswith(".doc"):
        file_types["doc"] += 1

    try:
        # Skip temporary files
        if file_lower.startswith("~$"):
            return file_types, processed_files, error_files

        # Define output path
        output_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.md")

        # Check for and remove existing original markdown file
        original_md_path = os.path.join(
            output_folder, f"{os.path.splitext(file)[0]}_original.md"
        )
        if os.path.exists(original_md_path):
            os.remove(original_md_path)

        # Process based on file type
        if file_lower.endswith(".pdf"):
            result = pdf_converter.pdf_to_text(input_path, output_format="md")
            if not result:
                raise Exception("Không thể chuyển đổi PDF sang Markdown")
        else:
            # Handle DOC/DOCX
            extract_func = (
                extract_text_from_docx
                if file_lower.endswith(".docx")
                else extract_text_from_doc
            )
            text = extract_func(input_path)
            if text:
                # Write directly to final output file
                with open(output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(text)
            else:
                raise Exception("Không thể trích xuất nội dung")

        logging.info(f"✅ Đã xử lý: {file}")
        processed_files.append(file)

        # Move processed files to a processed folder
        move_to_processed_folder(input_path, processed_folder)

        if os.path.dirname(input_path) == error_folder:
            logging.info(f"✅ Đã xử lý file từ thư mục error: {file}")

    except Exception as e:
        error_msg = f"❌ Lỗi xử lý {file}: {str(e)}"
        logging.error(error_msg)

        # Append to log with timestamp
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"## {time.strftime('%H:%M:%S')} - {file}\n")
            log.write(f"{str(e)}\n\n")

        move_to_error_folder(input_path, error_folder)
        if file not in error_files:
            error_files.append(file)

    return file_types, processed_files, error_files


def process_files(
    files: List[str], output_folder: str, error_folder: str, processed_folder: str
) -> Tuple[dict, List[str], List[str]]:
    """
    Process files and handle errors with visual progress tracking
    Returns:
        Tuple containing:
        - Dictionary of file types and counts
        - List of successfully processed files
        - List of error files
    """
    pdf_converter = PDFConverter()
    log_file = os.path.join(error_folder, "error_log.txt")
    file_types = {"pdf": 0, "doc": 0, "docx": 0}
    processed_files = []
    error_files = []

    # Clear previous log file content
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"# Error Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Khởi tạo batch info file để theo dõi tiến trình
    batch_info_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "batch_info.json"
    )

    # Lưu thông tin batch khi bắt đầu
    try:
        with open(batch_info_file, "w") as f:
            json.dump(
                {
                    "total_files": len(files),
                    "current_index": 0,
                    "start_time": time.time(),
                },
                f,
            )
    except Exception as e:
        logging.error(f"⚠️ Không thể lưu thông tin batch: {str(e)}")

    try:
        total_files = len(files)
        logging.info(f"Bắt đầu xử lý {total_files} files...")

        # Sử dụng tqdm để hiển thị tiến độ với ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as executor:
            try:
                # Tạo tất cả các future objects trước
                futures = {
                    executor.submit(
                        process_file,
                        input_path,
                        output_folder,
                        error_folder,
                        processed_folder,
                        pdf_converter,
                        log_file,
                    ): input_path
                    for input_path in files
                }

                # Sử dụng tqdm cùng với logging
                progress_bar = tqdm(
                    total=len(futures),
                    desc="Xử lý files",
                    unit="file",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )

                for i, future in enumerate(as_completed(futures)):
                    input_path = futures[future]
                    filename = os.path.basename(input_path)
                    try:
                        # Cập nhật thông tin batch
                        try:
                            with open(batch_info_file, "w") as f:
                                json.dump(
                                    {
                                        "total_files": total_files,
                                        "current_index": i,
                                        "current_file": filename,
                                        "processed": len(processed_files),
                                        "errors": len(error_files),
                                        "start_time": time.time(),
                                    },
                                    f,
                                )
                        except Exception:
                            pass

                        # Lấy kết quả xử lý
                        file_type, processed, errors = future.result()

                        # Cập nhật thống kê
                        for key in file_types:
                            file_types[key] += file_type.get(key, 0)
                        processed_files.extend(processed)
                        error_files.extend(errors)

                        # Cập nhật thông tin thanh tiến độ
                        progress_bar.set_postfix(
                            success=f"{len(processed_files)}/{i + 1}",
                            errors=len(error_files),
                        )
                        progress_bar.update(1)

                    except Exception as e:
                        logging.error(f"❌ Lỗi xử lý file {filename}: {str(e)}")
                        if filename not in error_files:
                            error_files.append(filename)

                        # Ghi lỗi vào log
                        with open(log_file, "a", encoding="utf-8") as log:
                            log.write(f"## {time.strftime('%H:%M:%S')} - {filename}\n")
                            log.write(f"{str(e)}\n\n")

                        # Di chuyển file lỗi vào thư mục error
                        try:
                            move_to_error_folder(input_path, error_folder)
                        except Exception as move_err:
                            logging.error(
                                f"Không thể di chuyển file lỗi: {str(move_err)}"
                            )

                        # Vẫn cập nhật thanh tiến độ khi gặp lỗi
                        progress_bar.update(1)
                progress_bar.close()

            except KeyboardInterrupt:
                # Dừng các futures đang chạy khi có Ctrl+C
                progress_bar.close()
                logging.warning("\n⚠️ Đang dừng xử lý do người dùng yêu cầu (Ctrl+C)...")

                # Lưu thông tin batch cuối
                try:
                    with open(batch_info_file, "w") as f:
                        json.dump(
                            {
                                "total_files": total_files,
                                "current_index": i if "i" in locals() else 0,
                                "processed": len(processed_files),
                                "errors": len(error_files),
                                "interrupted": True,
                                "last_update": time.time(),
                            },
                            f,
                        )
                except Exception:
                    pass

                logging.info(
                    "💾 Đã lưu thông tin xử lý. Quá trình xử lý PDF đang thực hiện sẽ tự lưu tiến độ."
                )

                # Cancel remaining futures
                for fut in futures:
                    if not fut.done() and not fut.cancelled():
                        fut.cancel()

                # Đợi các futures đang chạy hoàn thành hoặc bị cancel
                # Lưu ý: không thể cancel futures đang chạy, chỉ các futures chưa bắt đầu
                logging.info("⌛ Đang đợi các tiến trình hiện tại hoàn tất...")
                time.sleep(3)

        # Hiển thị kết quả cuối cùng
        success_rate = (
            len(processed_files) / total_files * 100 if total_files > 0 else 0
        )
        logging.info(
            f"✅ Hoàn thành: {len(processed_files)}/{total_files} files ({success_rate:.1f}%)"
        )

        if error_files:
            logging.warning(
                f"⚠️ Có {len(error_files)} files lỗi, xem chi tiết trong log"
            )
        logging.info("⌛ Đang đợi các tiến trình hiện tại hoàn tất...")
        time.sleep(3)

    except Exception as e:
        logging.error(f"❌ Lỗi trong quá trình xử lý files: {str(e)}")

    finally:
        # Make sure to cleanup
        try:
            pdf_converter.cleanup()
        except Exception:
            pass

        return file_types, processed_files, error_files


def create_api_json():
    """Create or update api.json file with Google API key"""
    try:
        # Get base directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        api_file = os.path.join(base_dir, "api.json")

        # Check if api.json exists and is valid
        if os.path.exists(api_file):
            try:
                with open(api_file, "r", encoding="utf-8") as f:
                    existing_config = json.load(f)
                    if existing_config.get("api_key"):
                        logging.info("\n✅ Đã tìm thấy API key")
                        use_existing = (
                            input("Bạn có muốn sử dụng API key hiện tại? (y/n): ")
                            .lower()
                            .strip()
                        )
                        if use_existing == "y":
                            return True
            except json.JSONDecodeError:
                logging.warning("\n⚠️ File api.json hiện tại không hợp lệ. Tạo mới...")
            except Exception as e:
                logging.warning(f"\n⚠️ Lỗi đọc file api.json: {str(e)}")

        # Get API key from user
        logging.info("\n📝 Cài đặt Google API Key")
        logging.info("1. Truy cập: https://makersuite.google.com/app/apikey")
        logging.info("2. Tạo API key mới hoặc sử dụng key có sẵn")
        api_key = input("\n🔑 Nhập API key của bạn: ").strip()

        if not api_key:
            logging.error("❌ API key không được để trống")
            return False

        # Create API configuration
        config = {"api_key": api_key}

        # Write to file
        with open(api_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        logging.info(f"\n✅ Đã lưu API key vào: {api_file}")
        return True

    except Exception as e:
        logging.error(f"\n❌ Lỗi tạo file api.json: {str(e)}")
        return False


def main():
    """Main execution flow"""
    try:
        # Setup logging
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_folder = os.path.join(base_dir, "data", "logs")
        logger = setup_logging(log_folder)

        logging.info("🚀 Chương trình bắt đầu")

        # 0. Check API configuration
        if not os.path.exists("api.json"):
            logging.warning("⚠️ Chưa có file cấu hình API")
            if not create_api_json():
                logging.error("❌ Không thể tiếp tục khi chưa có API key")
                return 1

        # Rest of your main function with logging instead of print
        logging.info("📂 Đang tạo thư mục...")
        input_folder, output_folder, error_folder, processed_folder = setup_folders()
        logging.info("✅ Đã tạo xong các thư mục cần thiết")

        # 1. Setup folders
        logging.info("\n📂 Đang tạo thư mục...")
        input_folder, output_folder, error_folder, processed_folder = setup_folders()
        logging.info("✅ Đã tạo xong các thư mục cần thiết")

        # 2. Check input files
        logging.info("\n🔍 Đang kiểm tra file...")
        # Fix this line to pass all required arguments
        valid_files, is_error_processing = check_input_files(
            input_folder, error_folder, output_folder, processed_folder
        )
        if not valid_files:
            logging.error("❌ Không tìm thấy file nào để xử lý")
            logging.info(f"ℹ️ Vui lòng thêm file vào thư mục: {input_folder}")
            return 1

        # 3. Process files
        source_folder = error_folder if is_error_processing else input_folder
        total_files = len(valid_files)
        logging.info(f"\n🔄 Bắt đầu xử lý {total_files} files từ {source_folder}...")

        # Get file statistics from process_files
        # You can either modify process_files to accept processed_folder or keep using the
        # hardcoded path in the process_files function
        file_types, processed_files, error_files = process_files(
            valid_files, output_folder, error_folder, processed_folder
        )

        logging.info("\n✅ Hoàn tất xử lý files")

        # Read error log and store details
        error_details = {}
        error_log_path = os.path.join(error_folder, "error_log.txt")
        if os.path.exists(error_log_path):
            with open(error_log_path, "r", encoding="utf-8") as log:
                for line in log:
                    if "❌ Lỗi xử lý" in line:
                        file_name = line.split("❌ Lỗi xử lý")[1].split(":")[0].strip()
                        error_msg = line.split(":", 1)[1].strip()
                        error_details[file_name] = error_msg

        # Print statistics
        logging.info("\n📁 TỔNG QUAN")
        logging.info(f"- Nguồn xử lý: {source_folder}")
        logging.info(f"- Tổng số file: {total_files}")

        logging.info("\n📊 PHÂN LOẠI")
        logging.info("- Định dạng file:")
        for ext, count in file_types.items():
            if count > 0:
                logging.info(f"  • {ext.upper()}: {count} files")

        print("\n📈 KẾT QUẢ")
        success_count = len(processed_files)
        error_count = len(error_files)
        print(
            f"- Thành công: {success_count} files ({success_count / total_files * 100:.1f}%)"
        )
        print(
            f"- Thất bại: {error_count} files ({error_count / total_files * 100:.1f}%)"
        )

        if error_files:
            print("\n❌ CHI TIẾT LỖI")
            for file in error_files:
                error_msg = error_details.get(file, "Không có thông tin lỗi")
                print(f"- {file}")
                print(f"  → {error_msg}")

        print("\n" + "=" * 50)
        if error_count > 0:
            logging.warning(f"⚠️ Xem chi tiết lỗi trong: {error_log_path}")
            logging.warning(f"📁 Files lỗi được chuyển vào: {error_folder}")
        else:
            logging.info("✅ Tất cả files đã được xử lý thành công!")

        return 0

    except Exception as e:
        logging.error(f"\n❌ Lỗi không xử lý được: {str(e)}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        logging.info("\n⚠️ Đã dừng chương trình.")
        exit_code = 0
    except Exception as e:
        logging.error(f"\n❌ Lỗi không xử lý được: {str(e)}")
        exit_code = 1
    finally:
        sys.exit(exit_code)
