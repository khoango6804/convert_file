import os
import argparse
from doc_converter import extract_text_from_doc, extract_text_from_docx
from file_utils import ensure_folder_exists, move_to_error_folder
from file_visualizer import visualize_data

def convert_all_docs(input_folder, output_folder, error_folder):
    """Chuyển đổi file Word và thống kê dữ liệu."""
    ensure_folder_exists(output_folder)
    ensure_folder_exists(error_folder)

    log_file = os.path.join(error_folder, "error_log.txt")

    with open(log_file, "w", encoding="utf-8") as log:
        for root, _, files in os.walk(input_folder):
            for file in files:
                input_path = os.path.join(root, file)

                if file.lower().startswith("~$"):  # Bỏ qua file tạm
                    continue

                if file.lower().endswith(".doc"):
                    output_path = os.path.join(output_folder, file.replace(".doc", ".txt"))
                    extract_func = extract_text_from_doc
                elif file.lower().endswith(".docx"):
                    output_path = os.path.join(output_folder, file.replace(".docx", ".txt"))
                    extract_func = extract_text_from_docx
                else:
                    continue  

                if os.path.exists(output_path):  
                    continue  

                try:
                    text = extract_func(input_path)
                    if text:
                        with open(output_path, "w", encoding="utf-8") as txt_file:
                            txt_file.write(text)
                    else:
                        raise Exception("Không thể trích xuất nội dung.")

                except Exception as e:
                    error_msg = f"Lỗi chuyển đổi {input_path}: {e}"
                    log.write(error_msg + "\n")
                    move_to_error_folder(input_path, error_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi file .doc và .docx sang .txt và visualize dữ liệu")
    parser.add_argument("input_folder", type=str, help="Thư mục chứa file Word")
    parser.add_argument("output_folder", type=str, help="Thư mục để lưu file .txt")
    parser.add_argument("error_folder", type=str, help="Thư mục để lưu file lỗi")

    args = parser.parse_args()

    convert_all_docs(args.input_folder, args.output_folder, args.error_folder)
    visualize_data(args.input_folder, args.output_folder, args.error_folder)
