import os
import sys
from typing import List, Tuple

from utils.doc_converter import extract_text_from_doc, extract_text_from_docx
from utils.file_utils import ensure_folder_exists, move_to_error_folder
from utils.file_visualizer import visualize_data
from utils.pdf_converter import PDFConverter


def setup_folders() -> Tuple[str, str, str]:
    """Setup required folders and return their paths"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders = {
        "input": os.path.join(base_dir, "data", "input"),
        "output": os.path.join(base_dir, "data", "output"),
        "error": os.path.join(base_dir, "data", "error"),
    }

    # Create folders if they don't exist
    for folder in folders.values():
        ensure_folder_exists(folder)

    return folders["input"], folders["output"], folders["error"]


def check_input_files(input_folder: str, error_folder: str) -> Tuple[List[str], bool]:
    """Check input and error folders for valid files"""
    input_files = []
    error_files = []

    # Check input folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith((".doc", ".docx", ".pdf")):
                input_files.append(os.path.join(root, file))

    # Check error folder
    for root, _, files in os.walk(error_folder):
        for file in files:
            if file.lower() != "error_log.txt" and file.lower().endswith(
                (".doc", ".docx", ".pdf")
            ):
                error_files.append(os.path.join(root, file))

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


def process_files(files: List[str], output_folder: str, error_folder: str) -> None:
    """Process files and handle errors"""
    pdf_converter = None
    log_file = os.path.join(error_folder, "error_log.txt")

    try:
        pdf_converter = PDFConverter()

        with open(log_file, "w", encoding="utf-8") as log:
            for input_path in files:
                file = os.path.basename(input_path)
                file_lower = file.lower()

                try:
                    # Skip temporary files
                    if file_lower.startswith("~$"):
                        continue

                    # Define output path and processing function
                    if file_lower.endswith(".pdf"):
                        output_path = os.path.join(
                            output_folder, f"{os.path.splitext(file)[0]}.txt"
                        )
                        result = pdf_converter.pdf_to_text(input_path)
                        if not result:
                            raise Exception("Không thể chuyển đổi PDF")
                    else:
                        # Handle DOC/DOCX
                        output_path = os.path.join(
                            output_folder,
                            file.replace(".docx", ".txt").replace(".doc", ".txt"),
                        )
                        extract_func = (
                            extract_text_from_docx
                            if file_lower.endswith(".docx")
                            else extract_text_from_doc
                        )

                        text = extract_func(input_path)
                        if text:
                            with open(output_path, "w", encoding="utf-8") as txt_file:
                                txt_file.write(text)
                        else:
                            raise Exception("Không thể trích xuất nội dung")

                    print(f"✅ Đã xử lý: {file}")

                except Exception as e:
                    error_msg = f"❌ Lỗi xử lý {file}: {str(e)}"
                    print(error_msg)
                    log.write(f"{error_msg}\n")
                    move_to_error_folder(input_path, error_folder)

    finally:
        if pdf_converter:
            pdf_converter.cleanup()


def main():
    """Main execution flow"""
    try:
        # 1. Setup folders
        input_folder, output_folder, error_folder = setup_folders()

        # 2. Check input files
        valid_files, is_error_processing = check_input_files(input_folder, error_folder)
        if not valid_files:
            return 1

        # 3. Process files
        source_folder = error_folder if is_error_processing else input_folder
        print(f"\n🔄 Đang xử lý {len(valid_files)} files từ {source_folder}...\n")
        process_files(valid_files, output_folder, error_folder)

        # 4. Generate visualization
        visualize_data(input_folder, output_folder, error_folder)

        return 0

    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
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
