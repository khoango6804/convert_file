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


def move_with_retry(source_path, dest_path, max_attempts=5, delay=1):
    """Move a file with retry if it's being used"""
    for attempt in range(max_attempts):
        try:
            shutil.move(source_path, dest_path)
            return True
        except PermissionError:
            if attempt < max_attempts - 1:
                print(
                    f"⚠️ File đang được sử dụng, thử lại sau {delay}s ({attempt + 1}/{max_attempts})..."
                )
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"⚠️ Không thể di chuyển file sau {max_attempts} lần thử")
                return False
        except Exception as e:
            print(f"⚠️ Lỗi di chuyển file: {str(e)}")
            return False


def move_to_processed_folder(source_path, processed_folder, remove_original=True):
    """Move or copy successfully processed file to the processed folder"""
    try:
        ensure_folder_exists(processed_folder)

        filename = os.path.basename(source_path)
        dest_path = os.path.join(processed_folder, filename)

        # Handle case when file already exists
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            dest_path = os.path.join(
                processed_folder, f"{base}_{int(time.time())}{ext}"
            )

        # Only move if source exists and is different from destination
        if os.path.exists(source_path) and source_path != dest_path:
            try:
                # Use this inside move_to_processed_folder
                if remove_original:
                    if move_with_retry(source_path, dest_path):
                        print(f"✅ Di chuyển file đã xử lý thành công đến: {dest_path}")
                    else:
                        # Fall back to copy if move fails
                        try:
                            shutil.copy2(source_path, dest_path)
                            print(
                                f"✅ Đã sao chép file đến: {dest_path} (không thể di chuyển)"
                            )
                        except Exception as copy_err:
                            print(f"⚠️ Không thể sao chép file: {str(copy_err)}")
                            return None
                else:
                    shutil.copy2(source_path, dest_path)
                    print(f"✅ Sao chép file đã xử lý thành công đến: {dest_path}")
            except PermissionError:
                # If file is still in use, try copy instead and schedule deletion
                print(f"⚠️ File đang được sử dụng: {source_path}")
                if remove_original:
                    try:
                        # Copy now, schedule deletion for later
                        shutil.copy2(source_path, dest_path)
                        print(f"✅ Đã sao chép file đến: {dest_path}")

                        # Schedule file for deletion after process exits
                        import atexit

                        def delete_later(path):
                            try:
                                if os.path.exists(path):
                                    os.remove(path)
                                    print(f"✅ Đã xóa file gốc: {path}")
                            except Exception as e:
                                print(
                                    f"⚠️ Không thể xóa file gốc: {path}, lỗi: {str(e)}"
                                )

                        atexit.register(delete_later, source_path)
                        return dest_path
                    except Exception as copy_err:
                        print(f"⚠️ Không thể sao chép file: {str(copy_err)}")
                        return None
                else:
                    # Just copy without deletion
                    try:
                        shutil.copy2(source_path, dest_path)
                        print(f"✅ Đã sao chép file đến: {dest_path}")
                        return dest_path
                    except Exception as copy_err:
                        print(f"⚠️ Không thể sao chép file: {str(copy_err)}")
                        return None
            except Exception as e:
                print(f"⚠️ Lỗi di chuyển/sao chép file: {str(e)}")
                return None
        return dest_path
    except Exception as e:
        print(f"❌ Không thể xử lý file đã hoàn thành: {str(e)}")
        return None


def is_already_processed(input_path: str, output_folder: str) -> bool:
    """
    Check if a file has already been processed by looking for its corresponding
    markdown file in the output folder

    Args:
        input_path: Path to the input file
        output_folder: Path to the output folder

    Returns:
        bool: True if file has already been processed, False otherwise
    """
    file_name = os.path.basename(input_path)
    base_name = os.path.splitext(file_name)[0]
    expected_output = os.path.join(output_folder, f"{base_name}.md")

    return os.path.exists(expected_output)
