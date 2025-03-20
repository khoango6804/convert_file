# PDF to Text Converter

Công cụ chuyển đổi tài liệu PDF thành văn bản có cấu trúc với sự hỗ trợ của AI.

## Tính năng chính

- 🧠 Sử dụng Google Gemini AI để phân tích và chuyển đổi tài liệu
- 📄 Hỗ trợ xử lý file PDF, DOC, DOCX
- 📊 Tự động tối ưu hóa xử lý theo cấu hình máy tính
- 🔍 Cải thiện chất lượng OCR thông qua tiền xử lý hình ảnh
- 📝 Xuất file markdown có định dạng
- ⏸️ Hỗ trợ lưu tiến trình và tiếp tục xử lý khi gặp sự cố
- 🔄 Xử lý hàng loạt file với khả năng theo dõi tiến trình chi tiết
- 💾 Tự động phân loại và lưu file đã xử lý

## Cài đặt

```bash
# Clone repository
git clone https://github.com/yourusername/convert_file.git

# Di chuyển đến thư mục dự án
cd convert_file

# Cài đặt các gói phụ thuộc
pip install -r requirements.txt
```

## Cấu hình

1. Đăng ký Google API Key tại <https://makersuite.google.com/app/apikey>
2. Chạy chương trình và nhập API key khi được yêu cầu

## Sử dụng

```bash
# Chạy chương trình với giao diện dòng lệnh
python main.py

# Hoặc chỉ định thư mục đầu vào
python main.py --input /path/to/files

# Để tiếp tục xử lý file PDF bị gián đoạn
python main.py --resume
```

## Cấu trúc thư mục

```
convert_file/
├─ data/
│  ├─ input/      # Thư mục chứa file cần xử lý
│  ├─ output/     # Thư mục lưu file đã chuyển đổi
│  ├─ processed/  # File gốc đã xử lý thành công
│  ├─ error/      # File có lỗi khi xử lý
│  └─ logs/       # File nhật ký
├─ utils/         # Các module tiện ích
│  ├─ pdf_converter.py
│  ├─ doc_converter.py
│  └─ ...
├─ main.py        # Điểm vào chính của chương trình
└─ README.md
```

## Yêu cầu hệ thống

- Python 3.8 hoặc cao hơn
- Kết nối internet để sử dụng Google Gemini API
- RAM tối thiểu 4GB (khuyến nghị 8GB để xử lý PDF dung lượng lớn)

## Tối ưu hóa hiệu suất

- Chương trình tự động điều chỉnh số lượng worker dựa trên số CPU và RAM
- Xử lý ưu tiên các file nhỏ và đơn giản trước để có kết quả nhanh chóng
- Sử dụng cache để tránh xử lý lặp lại nội dung

## Xử lý lỗi

- Tự động lưu tiến độ khi người dùng nhấn Ctrl+C
- Di chuyển file lỗi vào thư mục riêng và ghi lại thông tin lỗi chi tiết
- Tự động thử lại khi gặp lỗi API tạm thời

## Giấy phép

Dự án được cấp phép theo giấy phép MIT - xem file LICENSE để biết thêm chi tiết.

# File Deduplication System

An efficient tool for identifying and removing duplicate files using a multi-stage filtering approach.

## Features

- **Progressive filtering**: Size-based grouping, followed by filename and content comparisons
- **Resource optimization**: Memory-efficient chunked file processing and multithreading
- **Visual feedback**: Detailed progress bars showing current stage and completion percentage
- **Flexible handling**: Multiple strategies for managing duplicates
- **Safety mechanisms**: Interactive confirmation before deletion

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/file-dedup-system.git
cd file-dedup-system

# Install required packages
pip install -r requirements.txt
```

## Usage

Basic usage:

```bash
python check_dup.py /path/to/directory
```

Or with named parameter:

```bash
python check_dup.py --directory /path/to/directory
```

Advanced options:

```bash
python check_dup.py /path/to/directory --recursive --action=interactive --keep=newest
```

### Resuming Interrupted Scans

If the scanning process gets interrupted (especially for large directories), you can resume from where you left off:

```bash
python check_dup.py /path/to/directory --resume
```

This will pick up from the last saved checkpoint for that directory.

### Command-line Options

- `directory` or `--directory`: Path to the directory to scan for duplicates
- `-r, --recursive`: Scan directories recursively (default: False)
- `-d, --max-depth`: Maximum recursion depth for directory traversal
- `-e, --exclude`: File extensions to exclude (e.g., `-e jpg png`)
- `-i, --include`: Only include these file extensions (e.g., `-i mp3 wav`)
- `-t, --threads`: Number of threads to use for parallel processing
- `-a, --action`: Action to take with duplicates (`report`, `delete`, or `interactive`)
- `-k, --keep`: Strategy for selecting which duplicate to keep (`newest`, `oldest`, or `first_found`)
- `-s, --symlink`: Create symlinks to kept files instead of deleting duplicates
- `-c, --chunk-size`: Chunk size for file reading in bytes
- `--resume`: Resume from a previous checkpoint if available
- `--clear-checkpoints`: Clear all saved checkpoints before starting

## Example Workflows

### Generate a Report of Duplicates

```bash
python check_dup.py /path/to/photos -r -a report
```

### Interactive Duplicate Management

```bash
python check_dup.py /path/to/documents -r -a interactive -k oldest
```

### Automatic Deletion with Symlinks

```bash
python check_dup.py /path/to/music -r -a delete -s -i mp3 wav flac
```

## System Requirements

- Python 3.8 or higher
- Sufficient disk space for temporary files
- Adequate memory for processing large file sets

## License

[MIT License](LICENSE)
