import time
import os
import sys
import json
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.doc_converter import extract_text_from_doc, extract_text_from_docx
from utils.file_utils import (
    ensure_folder_exists,
    move_to_error_folder,
    move_to_processed_folder,
    is_already_processed,
)
from utils.pdf_converter import PDFConverter


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
                    print(f"⏭️ Đã tìm thấy file đã xử lý: {file}")
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
        print(
            f"⏭️ Đã chuyển {len(already_processed)} file đã xử lý trước đó sang thư mục processed"
        )

    # If input is empty but error has files, process error folder
    if not input_files and error_files:
        print(
            f"⚠️ Thư mục input trống. Tìm thấy {len(error_files)} file trong thư mục error."
        )
        print("🔄 Chuyển sang xử lý files từ thư mục error...")
        return error_files, True

    # If input has files, process those
    if input_files:
        return input_files, False

    # If both are empty
    print("⚠️ Không tìm thấy file .doc, .docx hoặc .pdf nào để xử lý")
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

        print(f"✅ Đã xử lý: {file}")
        processed_files.append(file)

        # Move processed files to a processed folder
        move_to_processed_folder(input_path, processed_folder)

        if os.path.dirname(input_path) == error_folder:
            print(f"✅ Đã xử lý file từ thư mục error: {file}")

    except Exception as e:
        error_msg = f"❌ Lỗi xử lý {file}: {str(e)}"
        print(error_msg)

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
    Process files and handle errors
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

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                process_file,
                input_path,
                output_folder,
                error_folder,
                processed_folder,
                pdf_converter,
                log_file,
            )
            for input_path in files
        ]
        for future in as_completed(futures):
            try:
                file_type, processed, errors = future.result()
                for key in file_types:
                    file_types[key] += file_type.get(key, 0)
                processed_files.extend(processed)
                error_files.extend(errors)
            except Exception as e:
                print(f"❌ Lỗi xử lý file: {str(e)}")

    pdf_converter.cleanup()
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
                        print("\n✅ Đã tìm thấy API key")
                        use_existing = (
                            input("Bạn có muốn sử dụng API key hiện tại? (y/n): ")
                            .lower()
                            .strip()
                        )
                        if use_existing == "y":
                            return True
            except json.JSONDecodeError:
                print("\n⚠️ File api.json hiện tại không hợp lệ. Tạo mới...")
            except Exception as e:
                print(f"\n⚠️ Lỗi đọc file api.json: {str(e)}")

        # Get API key from user
        print("\n📝 Cài đặt Google API Key")
        print("1. Truy cập: https://makersuite.google.com/app/apikey")
        print("2. Tạo API key mới hoặc sử dụng key có sẵn")
        api_key = input("\n🔑 Nhập API key của bạn: ").strip()

        if not api_key:
            print("❌ API key không được để trống")
            return False

        # Create API configuration
        config = {"api_key": api_key}

        # Write to file
        with open(api_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        print(f"\n✅ Đã lưu API key vào: {api_file}")
        return True

    except Exception as e:
        print(f"\n❌ Lỗi tạo file api.json: {str(e)}")
        return False


def main():
    """Main execution flow"""
    try:
        # 0. Check API configuration
        if not os.path.exists("api.json"):
            print("\n⚠️ Chưa có file cấu hình API")
            if not create_api_json():
                print("❌ Không thể tiếp tục khi chưa có API key")
                return 1

        # 1. Setup folders
        print("\n📂 Đang tạo thư mục...")
        input_folder, output_folder, error_folder, processed_folder = setup_folders()
        print("✅ Đã tạo xong các thư mục cần thiết")

        # 2. Check input files
        print("\n🔍 Đang kiểm tra file...")
        # Fix this line to pass all required arguments
        valid_files, is_error_processing = check_input_files(
            input_folder, error_folder, output_folder, processed_folder
        )
        if not valid_files:
            print("❌ Không tìm thấy file nào để xử lý")
            print(f"ℹ️ Vui lòng thêm file vào thư mục: {input_folder}")
            return 1

        # 3. Process files
        source_folder = error_folder if is_error_processing else input_folder
        total_files = len(valid_files)
        print(f"\n🔄 Bắt đầu xử lý {total_files} files từ {source_folder}...")

        # Get file statistics from process_files
        # You can either modify process_files to accept processed_folder or keep using the
        # hardcoded path in the process_files function
        file_types, processed_files, error_files = process_files(
            valid_files, output_folder, error_folder, processed_folder
        )

        print("\n✅ Hoàn tất xử lý files")

        # 4. Generate report
        print("\n" + "=" * 50)
        print("📊 BÁO CÁO XỬ LÝ FILES")
        print("=" * 50)

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
        print("\n📁 TỔNG QUAN")
        print(f"- Nguồn xử lý: {source_folder}")
        print(f"- Tổng số file: {total_files}")

        print("\n📊 PHÂN LOẠI")
        print("- Định dạng file:")
        for ext, count in file_types.items():
            if count > 0:
                print(f"  • {ext.upper()}: {count} files")

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
            print(f"⚠️ Xem chi tiết lỗi trong: {error_log_path}")
            print(f"📁 Files lỗi được chuyển vào: {error_folder}")
        else:
            print("✅ Tất cả files đã được xử lý thành công!")

        return 0

    except Exception as e:
        print(f"\n❌ Lỗi không xử lý được: {str(e)}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n⚠️ Đã dừng chương trình.")
        exit_code = 0
    except Exception as e:
        print(f"\n❌ Lỗi không xử lý được: {str(e)}")
        exit_code = 1
    finally:
        sys.exit(exit_code)
