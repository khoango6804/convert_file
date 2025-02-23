import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def count_files(folder_path, extensions=[".doc", ".docx", ".pdf"]):
    """Đếm số lượng file theo phần mở rộng trong một thư mục."""
    file_counts = {ext: 0 for ext in extensions}

    for root, _, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in file_counts:
                file_counts[ext] += 1

    return file_counts

def count_duplicate_files(folder_path):
    """Đếm số file trùng lặp theo tên (không phân biệt hoa thường)."""
    file_counter = Counter()
    duplicate_files = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_name = file.lower()  
            file_counter[file_name] += 1

    duplicates = {file: count for file, count in file_counter.items() if count > 1}

    return duplicates

def visualize_data(input_folder, output_folder, error_folder):
    """Vẽ biểu đồ thống kê file tổng, file lỗi, file trùng."""
    total_counts = count_files(input_folder)
    error_counts = count_files(error_folder)
    duplicate_counts = count_duplicate_files(output_folder)

    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Vẽ biểu đồ
    plt.subplot(1, 3, 1)
    plt.bar(total_counts.keys(), total_counts.values(), color="blue")
    plt.title("Số lượng file tổng")
    plt.xlabel("Loại file")
    plt.ylabel("Số lượng")

    plt.subplot(1, 3, 2)
    plt.bar(error_counts.keys(), error_counts.values(), color="red")
    plt.title("Số lượng file lỗi")
    plt.xlabel("Loại file")

    plt.subplot(1, 3, 3)
    plt.bar(duplicate_counts.keys(), duplicate_counts.values(), color="orange")
    plt.title("Số lượng file trùng lặp")
    plt.xlabel("Tên file")
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.show()
