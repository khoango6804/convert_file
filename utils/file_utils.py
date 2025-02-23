import os
import shutil

def ensure_folder_exists(folder_path):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def move_to_error_folder(input_path, error_folder):
    """Di chuyển file lỗi vào thư mục lỗi"""
    ensure_folder_exists(error_folder)
    try:
        error_path = os.path.join(error_folder, os.path.basename(input_path))
        shutil.move(input_path, error_path)
        print(f"⚠️ Di chuyển file lỗi đến: {error_path}")
    except PermissionError:
        print(f"🚫 Không thể di chuyển file (PermissionError): {input_path}")
