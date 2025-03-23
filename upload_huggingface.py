import os
import glob
import argparse
import logging
import time
from datasets import Dataset, Features, Value
from huggingface_hub import HfApi, login
from tqdm import tqdm

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def setup_paths(input_dir=None, input_file=None):
    """Thiết lập các đường dẫn cần thiết"""
    if input_file:
        # Sử dụng file được chỉ định
        if not os.path.isfile(input_file):
            raise ValueError(f"File {input_file} không tồn tại!")
        return os.path.abspath(input_file), True

    if input_dir:
        # Sử dụng thư mục được chỉ định
        txts_dir = os.path.abspath(input_dir)
    else:
        # Sử dụng thư mục mặc định
        base_dir = os.path.dirname(os.path.abspath(__file__))
        txts_dir = os.path.join(base_dir, "data", "txts")

    if not os.path.exists(txts_dir):
        raise ValueError(f"Thư mục {txts_dir} không tồn tại!")

    return txts_dir, False


def authenticate(token=None):
    """Đăng nhập vào HuggingFace Hub"""
    if token is None:
        # Kiểm tra token trong biến môi trường
        token = os.environ.get("HF_TOKEN")

    if not token:
        # Nếu không có token, yêu cầu người dùng nhập
        token = input("Nhập HuggingFace API Token của bạn: ").strip()

    try:
        login(token=token)
        logging.info("✅ Đăng nhập HuggingFace thành công!")
        return token
    except Exception as e:
        raise ValueError(f"Đăng nhập thất bại: {str(e)}")


def read_text_files(txts_dir, file_pattern="*.txt"):
    """Đọc tất cả file txt và chuẩn bị dataset"""
    text_files = glob.glob(os.path.join(txts_dir, file_pattern))
    if not text_files:
        raise ValueError(
            f"Không tìm thấy file nào với mẫu {file_pattern} trong {txts_dir}"
        )

    logging.info(f"🔍 Đã tìm thấy {len(text_files)} file")

    documents = []
    file_names = []
    file_sizes = []

    # Đọc từng file với thanh tiến độ
    for txt_file in tqdm(text_files, desc="Đọc file", unit="file"):
        try:
            file_name = os.path.basename(txt_file)
            file_size = os.path.getsize(txt_file)

            with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            documents.append(text)
            file_names.append(file_name)
            file_sizes.append(file_size)

        except Exception as e:
            logging.warning(f"⚠️ Không thể đọc file {txt_file}: {str(e)}")

    # Tạo dataset từ các tài liệu đã đọc
    dataset_dict = {
        "text": documents,
        "file_name": file_names,
        "file_size": file_sizes,
    }

    return dataset_dict


def read_single_file(file_path):
    """Đọc một file duy nhất và chuẩn bị dataset"""
    try:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        logging.info(f"🔍 Đang đọc file: {file_name}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        # Tạo dataset từ file đã đọc
        dataset_dict = {
            "text": [text],
            "file_name": [file_name],
            "file_size": [file_size],
        }

        logging.info(f"✅ Đã đọc file {file_name} ({file_size / 1024:.2f} KB)")
        return dataset_dict

    except Exception as e:
        raise ValueError(f"Không thể đọc file {file_path}: {str(e)}")


def create_and_push_dataset(dataset_dict, repo_name, token, description=None):
    """Tạo dataset và đẩy lên HuggingFace Hub"""
    try:
        # Tạo dataset từ dict
        features = Features(
            {
                "text": Value("string"),
                "file_name": Value("string"),
                "file_size": Value("int64"),
            }
        )

        dataset = Dataset.from_dict(dataset_dict, features=features)
        logging.info(f"✅ Đã tạo dataset với {len(dataset)} mẫu")

        # Tính kích thước tổng của dataset
        total_size_bytes = sum(dataset_dict["file_size"])
        total_size_mb = total_size_bytes / (1024 * 1024)
        logging.info(f"📊 Tổng kích thước: {total_size_mb:.2f} MB")

        # Push lên HuggingFace
        logging.info(f"🔄 Đang upload lên {repo_name}...")

        # Push dataset lên hub
        dataset.push_to_hub(
            repo_id=repo_name,
            token=token,
            private=True,  # Set to False nếu muốn dataset công khai
        )

        logging.info(
            f"✅ Upload thành công: https://huggingface.co/datasets/{repo_name}"
        )

        # Thêm thông tin README.md
        api = HfApi()

        custom_description = (
            description
            or f"Dataset bao gồm {len(dataset)} tài liệu văn bản được chuyển đổi từ định dạng Markdown sang text thuần túy."
        )

        readme_content = f"""
        # Dataset: {repo_name}
        
        {custom_description}
        
        ## Thông tin:
        
        - Số lượng tài liệu: {len(dataset)}
        - Tổng kích thước: {total_size_mb:.2f} MB
        - Ngày tạo: {time.strftime("%Y-%m-%d")}
        
        ## Cấu trúc:
        
        ```python
        DatasetDict({{
            'train': Dataset({{
                features: ['text', 'file_name', 'file_size'],
                num_rows: {len(dataset)}
            }})
        }})
        ```
        
        ## Sử dụng:
        
        ```python
        from datasets import load_dataset
        
        dataset = load_dataset("{repo_name}")
        for i, example in enumerate(dataset["train"]):
            print(f"Document {{i + 1}}:")
            print(example["text"][:500] + "...")  # Hiển thị 500 ký tự đầu tiên
            print("-" * 50)
        ```
        """

        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            repo_type="dataset",
            token=token,
        )

        return True

    except Exception as e:
        logging.error(f"❌ Upload thất bại: {str(e)}")
        return False


def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(
        description="Upload files lên HuggingFace Datasets"
    )
    parser.add_argument(
        "--repo", type=str, help="Tên repository trên HuggingFace (username/repo-name)"
    )
    parser.add_argument("--token", type=str, help="HuggingFace API Token")
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Thư mục chứa các file cần upload (mặc định: ./data/txts)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Đường dẫn đến một file duy nhất để upload",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.txt",
        help="Mẫu tên file cần tìm (mặc định: *.txt)",
    )
    parser.add_argument("--description", type=str, help="Mô tả dataset cho README")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Đặt dataset là công khai (mặc định: riêng tư)",
    )
    args = parser.parse_args()

    start_time = time.time()

    try:
        # Kiểm tra nếu cả hai đối số input-dir và input-file được cung cấp
        if args.input_dir and args.input_file:
            raise ValueError(
                "Chỉ sử dụng một trong hai tùy chọn --input-dir hoặc --input-file, không sử dụng cả hai."
            )

        # Thiết lập đường dẫn
        input_path, is_single_file = setup_paths(args.input_dir, args.input_file)

        # Xác định tên repository
        if not args.repo:
            default_name = f"text-dataset-{time.strftime('%Y%m%d')}"
            repo_name = (
                input(f"Nhập tên repository (mặc định: {default_name}): ").strip()
                or default_name
            )
        else:
            repo_name = args.repo

        # Đảm bảo định dạng username/repo-name
        if "/" not in repo_name:
            username = input("Nhập username HuggingFace của bạn: ").strip()
            repo_name = f"{username}/{repo_name}"

        # Xác thực
        token = authenticate(args.token)

        # Đọc dữ liệu (một file hoặc nhiều file)
        if is_single_file:
            logging.info(f"📄 Đang đọc file: {input_path}")
            dataset_dict = read_single_file(input_path)
        else:
            logging.info(f"📂 Đang đọc file từ {input_path} với mẫu {args.pattern}")
            dataset_dict = read_text_files(input_path, args.pattern)

        # Tạo và đẩy dataset
        success = create_and_push_dataset(
            dataset_dict, repo_name, token, args.description
        )

        # Thông báo kết quả
        if success:
            elapsed_time = time.time() - start_time
            logging.info(f"⏱️ Tổng thời gian: {elapsed_time:.2f} giây")
            logging.info(
                f"🔗 Dataset có thể truy cập tại: https://huggingface.co/datasets/{repo_name}"
            )
        else:
            logging.error("❌ Quá trình upload không thành công")

    except Exception as e:
        logging.error(f"❌ Lỗi: {str(e)}")


if __name__ == "__main__":
    main()
