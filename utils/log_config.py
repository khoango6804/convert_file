import logging
import os
import sys
from datetime import datetime
from tqdm import tqdm as tqdm_module  # Đã sửa tên import

# Tắt tất cả logger từ Google Generative AI trước khi làm bất cứ điều gì khác
for name in logging.root.manager.loggerDict:
    if name.startswith("google"):
        logging.getLogger(name).setLevel(logging.ERROR)
        # Xóa hết handler của các logger này để đảm bảo không ghi log
        logger = logging.getLogger(name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Đảm bảo logger không truyền log lên logger cha
        logger.propagate = False


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that works with tqdm progress bars"""

    def emit(self, record):
        # Lọc thông báo AFC ngay tại đây
        if "AFC" in record.getMessage():
            return

        try:
            msg = self.format(record)
            # Sử dụng tqdm_module.write thay vì tqdm.tqdm.write
            tqdm_module.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class AFCFilter(logging.Filter):
    """Lọc thông báo AFC từ Google API"""

    def filter(self, record):
        # Trả về False để loại bỏ thông báo có chứa 'AFC'
        return "AFC" not in record.getMessage()


def setup_logging(log_folder=None, level=logging.INFO):
    """
    Setup logging configuration
    Args:
        log_folder: Folder to store log files
        level: Logging level
    """
    # Lọc verbose logging từ Google Generative AI
    logging.getLogger("google").setLevel(logging.ERROR)
    logging.getLogger("google.generativeai").setLevel(logging.ERROR)
    logging.getLogger("google.generativeai.types").setLevel(logging.ERROR)
    logging.getLogger("google.auth").setLevel(logging.ERROR)

    # Create base formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Thêm filter để lọc thông báo AFC
    afc_filter = AFCFilter()
    root_logger.addFilter(afc_filter)

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with tqdm compatibility
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    console_handler.addFilter(afc_filter)  # Thêm filter ở cả handler level
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_folder:
        os.makedirs(log_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_folder, f"convert_{timestamp}.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        file_handler.addFilter(afc_filter)  # Thêm filter ở cả handler level
        root_logger.addHandler(file_handler)

        logging.info(f"Log file: {log_file}")

    return root_logger
