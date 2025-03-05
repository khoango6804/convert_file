import logging
import os
import sys
from datetime import datetime
from tqdm import tqdm as tqdm_module  # Đổi tên khi import để tránh nhầm lẫn


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that works with tqdm progress bars"""

    def emit(self, record):
        try:
            msg = self.format(record)
            # Sử dụng tqdm_module.write thay vì tqdm.tqdm.write
            tqdm_module.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(log_folder=None, level=logging.INFO):
    """
    Setup logging configuration
    Args:
        log_folder: Folder to store log files
        level: Logging level
    """
    # Suppress verbose logging from Google Generative AI library
    logging.getLogger("google.generativeai").setLevel(logging.ERROR)
    logging.getLogger("google.generativeai.types").setLevel(logging.ERROR)

    # Create base formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler with tqdm compatibility
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_folder:
        os.makedirs(log_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_folder, f"convert_{timestamp}.log")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

        logging.info(f"Log file: {log_file}")

    return root_logger
