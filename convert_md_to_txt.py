import os
import glob
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def setup_folders():
    """Khởi tạo các thư mục cần thiết"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folders = {
        "output_md": os.path.join(base_dir, "data", "output"),
        "output_txt": os.path.join(base_dir, "data", "txts"),  # Đổi tên thư mục output
    }

    # Tạo thư mục nếu chưa tồn tại
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    return folders


def clean_markdown(text):
    """Làm sạch nội dung markdown để chuyển thành text thuần túy"""
    # Loại bỏ các header markdown (#, ##, ###)
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)

    # Loại bỏ định dạng in đậm và in nghiêng
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)

    # Loại bỏ định dạng code
    text = re.sub(r"`(.*?)`", r"\1", text)

    # Xử lý code blocks
    text = re.sub(r"```(?:.*?)\n(.*?)```", r"\1", text, flags=re.DOTALL)

    # Loại bỏ định dạng link [text](url)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)

    # Loại bỏ dấu > của blockquote
    text = re.sub(r"^>\s+", "", text, flags=re.MULTILINE)

    # Loại bỏ định dạng danh sách
    text = re.sub(r"^\s*[\-\*]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Xóa các kí hiệu markdown khác
    text = re.sub(r"^\s*---\s*$", "", text, flags=re.MULTILINE)

    # Xử lý các bảng markdown
    lines = text.split("\n")
    result_lines = []
    in_table = False
    header_cells = []

    for line in lines:
        if "|" in line and ("-+-" in line or "---" in line or ":---" in line):
            in_table = True
            continue

        if in_table and "|" in line:
            # Chuyển đổi hàng bảng thành text đơn giản
            cells = [cell.strip() for cell in line.split("|")]
            cells = [cell for cell in cells if cell]  # Remove empty cells

            # Nếu đây là hàng đầu tiên sau header, lưu lại làm header
            if not header_cells:
                header_cells = cells.copy()
                result_lines.append(" | ".join(header_cells))
            else:
                result_lines.append(" | ".join(cells))
        else:
            if in_table:
                in_table = False
                header_cells = []
            result_lines.append(line)

    # Kết hợp lại các dòng và loại bỏ dòng trống liên tiếp
    text = "\n".join(result_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def convert_file(input_file, output_folder):
    """Chuyển đổi một file markdown sang text"""
    try:
        filename = os.path.basename(input_file)
        name_without_ext = os.path.splitext(filename)[0]
        output_file = os.path.join(output_folder, f"{name_without_ext}.txt")

        # Kiểm tra kích thước file để xác định phương pháp xử lý
        file_size = os.path.getsize(input_file)

        if file_size > 10 * 1024 * 1024:  # > 10MB
            # Xử lý file lớn dòng-theo-dòng
            try:
                with (
                    open(input_file, "r", encoding="utf-8") as infile,
                    open(output_file, "w", encoding="utf-8") as outfile,
                ):
                    for chunk in read_in_chunks(infile):
                        cleaned_chunk = clean_markdown(chunk)
                        outfile.write(cleaned_chunk)

                return output_file, file_size, None
            except UnicodeDecodeError:
                # Thử lại với encoding khác
                with (
                    open(input_file, "r", encoding="latin-1") as infile,
                    open(output_file, "w", encoding="utf-8") as outfile,
                ):
                    for chunk in read_in_chunks(infile):
                        cleaned_chunk = clean_markdown(chunk)
                        outfile.write(cleaned_chunk)

                return output_file, file_size, None
        else:
            # Xử lý file nhỏ
            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Thử với encoding khác nếu utf-8 thất bại
                with open(input_file, "r", encoding="latin-1") as f:
                    content = f.read()

            # Làm sạch content
            cleaned_content = clean_markdown(content)

            # Ghi ra file text
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(cleaned_content)

            return output_file, len(cleaned_content), None
    except Exception as e:
        return None, 0, f"Lỗi xử lý file {os.path.basename(input_file)}: {str(e)}"


def read_in_chunks(file_object, chunk_size=1024):
    """Đọc file theo từng đoạn để xử lý file lớn hiệu quả"""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data


def get_optimal_worker_count():
    """Xác định số worker tối ưu dựa trên cấu hình hệ thống"""
    import multiprocessing

    # Lấy số CPU cores
    cpu_count = multiprocessing.cpu_count()

    # Kiểm tra RAM khả dụng (nếu trên Windows)
    ram_gb = None
    try:
        if os.name == "nt":  # Windows
            import psutil

            ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass  # psutil không khả dụng, bỏ qua kiểm tra RAM

    # Đây là tác vụ I/O-bound, nên có thể dùng nhiều worker hơn số core
    if cpu_count <= 2:
        workers = 2  # Ít nhất 2 worker
    else:
        workers = cpu_count + 2  # Nhiều hơn số core vì chủ yếu là I/O

    # Điều chỉnh dựa trên RAM (nếu có thông tin)
    if ram_gb is not None and ram_gb < 4:  # Máy có ít RAM
        workers = min(workers, 4)  # Giới hạn số worker

    return workers


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển đổi file MD sang TXT riêng biệt"
    )
    parser.add_argument(
        "--override", action="store_true", help="Ghi đè lên các file TXT đã tồn tại"
    )
    args = parser.parse_args()

    start_time = time.time()
    folders = setup_folders()

    # Lấy danh sách file markdown
    md_files = glob.glob(os.path.join(folders["output_md"], "*.md"))

    if not md_files:
        logging.warning("❌ Không tìm thấy file markdown nào trong thư mục output!")
        return

    logging.info(f"🔍 Tìm thấy {len(md_files)} file markdown cần chuyển đổi")

    # Lọc các file đã tồn tại (trừ khi có flag override)
    if not args.override:
        original_count = len(md_files)
        md_files = [
            f
            for f in md_files
            if not os.path.exists(
                os.path.join(
                    folders["output_txt"],
                    os.path.splitext(os.path.basename(f))[0] + ".txt",
                )
            )
        ]
        skipped = original_count - len(md_files)
        if skipped > 0:
            logging.info(
                f"⏩ Bỏ qua {skipped} file đã được chuyển đổi trước đó (sử dụng --override để ghi đè)"
            )

    if not md_files:
        logging.info("✅ Tất cả các file đã được chuyển đổi trước đó!")
        return

    # Xác định số lượng worker tối ưu
    worker_count = get_optimal_worker_count()
    logging.info(f"🖥️ Sử dụng {worker_count} worker threads")

    # Khởi tạo thanh tiến độ
    successful_files = 0
    failed_files = 0
    total_chars = 0

    # Xử lý đa luồng với ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        # Tạo các futures cho mỗi file cần xử lý
        futures = {
            executor.submit(convert_file, md_file, folders["output_txt"]): md_file
            for md_file in md_files
        }

        # Sử dụng tqdm để hiển thị tiến độ
        with tqdm(
            total=len(md_files), desc="Chuyển đổi file", unit="file"
        ) as progress_bar:
            for future in as_completed(futures):
                md_file = futures[future]
                try:
                    result, chars, error = future.result()
                    if result:
                        successful_files += 1
                        total_chars += chars
                        progress_bar.set_postfix(
                            success=f"{successful_files}/{successful_files + failed_files}"
                        )
                    else:
                        failed_files += 1
                        logging.error(error)
                except Exception as e:
                    failed_files += 1
                    logging.error(f"❌ Lỗi xử lý {os.path.basename(md_file)}: {str(e)}")
                finally:
                    progress_bar.update(1)

    # Hiển thị kết quả
    elapsed_time = time.time() - start_time
    logging.info(
        f"✅ Chuyển đổi hoàn tất: {successful_files} thành công, {failed_files} thất bại"
    )
    logging.info(f"📊 Tổng số ký tự đã xử lý: {total_chars:,}")
    logging.info(f"⏱️ Thời gian thực hiện: {elapsed_time:.2f} giây")
    logging.info(f"📁 Các file txt đã được lưu trong: {folders['output_txt']}")


if __name__ == "__main__":
    main()
