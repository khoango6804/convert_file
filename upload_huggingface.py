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


def setup_paths():
    """Thiết lập các đường dẫn cần thiết"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    txts_dir = os.path.join(base_dir, "data", "txts")

    if not os.path.exists(txts_dir):
        raise ValueError(f"Thư mục {txts_dir} không tồn tại!")

    return txts_dir


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


def read_text_files(txts_dir):
    """Đọc tất cả file txt và chuẩn bị dataset"""
    text_files = glob.glob(os.path.join(txts_dir, "*.txt"))
    if not text_files:
        raise ValueError(f"Không tìm thấy file txt nào trong {txts_dir}")

    logging.info(f"🔍 Đã tìm thấy {len(text_files)} file txt")

    documents = []
    file_names = []
    file_sizes = []

    # Đọc từng file với thanh tiến độ
    for txt_file in tqdm(text_files, desc="Đọc file", unit="file"):
        try:
            file_name = os.path.basename(txt_file)
            file_size = os.path.getsize(txt_file)

            with open(txt_file, "r", encoding="utf-8") as f:
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


def create_and_push_dataset(dataset_dict, repo_name, token):
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
        readme_content = f"""
        # Dataset: {repo_name}
        
        Dataset bao gồm {len(dataset)} tài liệu văn bản được chuyển đổi từ định dạng Markdown sang text thuần túy.
        
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
        description="Upload txt files lên HuggingFace Datasets"
    )
    parser.add_argument(
        "--repo", type=str, help="Tên repository trên HuggingFace (username/repo-name)"
    )
    parser.add_argument("--token", type=str, help="HuggingFace API Token")
    args = parser.parse_args()

    start_time = time.time()

    try:
        # Thiết lập đường dẫn
        txts_dir = setup_paths()

        # Xác định tên repository
        if not args.repo:
            default_name = f"legal-documents-{time.strftime('%Y%m%d')}"
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

        # Đọc các file
        logging.info(f"📂 Đang đọc file từ {txts_dir}")
        dataset_dict = read_text_files(txts_dir)

        # Tạo và đẩy dataset
        success = create_and_push_dataset(dataset_dict, repo_name, token)

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
