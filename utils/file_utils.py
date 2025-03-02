import os
import shutil
import time

def ensure_folder_exists(folder_path):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def move_to_error_folder(source_path, error_folder):
    """Move file to error folder"""
    try:
        # Fix the typo in the path (Documennts → Documents)
        error_folder = error_folder.replace("Documennts", "Documents")
        
        filename = os.path.basename(source_path)
        dest_path = os.path.join(error_folder, filename)
        
        # Handle case when file already exists
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            dest_path = os.path.join(error_folder, f"{base}_{int(time.time())}{ext}")
            
        # Only move if source exists and is different from destination
        if os.path.exists(source_path) and source_path != dest_path:
            shutil.copy2(source_path, dest_path)
            print(f"⚠️ Di chuyển file lỗi đến: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"❌ Không thể di chuyển file lỗi: {str(e)}")
        return None
