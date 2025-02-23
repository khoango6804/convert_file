# 📝 Word to TXT Converter & Data Visualization

## 📌 Giới thiệu

Chương trình này giúp **chuyển đổi file Word (`.doc` & `.docx`) sang `.txt`**,
quản lý file lỗi, **thống kê số lượng file**, và **visualize dữ liệu** bằng biểu đồ.

## 🛠 Tính năng chính

✅ **Chuyển đổi file Word (.doc, .docx) sang TXT**  
✅ **Quản lý file lỗi**, tự động di chuyển vào thư mục riêng  
✅ **Thống kê số lượng file, file lỗi, file trùng**  
✅ **Visualize dữ liệu bằng biểu đồ Matplotlib & Seaborn**  

## 📂 Cấu trúc thư mục

## 📥 Cài đặt

**1️⃣ Cài đặt Python** (Nếu chưa có)  
Tải và cài đặt Python từ [python.org](https://www.python.org/downloads/)

**2️⃣ Cài đặt các thư viện cần thiết**  
Chạy lệnh sau trong terminal/cmd:

```sh
pip install -r requirements.txt

🚀 Cách chạy chương trình
Chạy lệnh sau để chuyển đổi file và visualize dữ liệu:
ví dụ:
python main.py "K:\downloads" "K:\output_txt" "K:\errors"

📌 Tham số:
K:\downloads → Thư mục chứa file Word
K:\output_txt → Thư mục để lưu file TXT
K:\errors → Thư mục chứa file lỗi

📊 Output Visualization
Sau khi chạy, chương trình sẽ hiển thị 3 biểu đồ thống kê:
Tổng số file theo định dạng
Số lượng file lỗi
Danh sách file trùng lặp
📌 Ví dụ biểu đồ:

❓ Lỗi phổ biến & Cách khắc phục
Lỗi Nguyên nhân Cách sửa
No module named 'win32com' Chưa cài pywin32 Chạy pip install pywin32
FileNotFoundError Đường dẫn sai Kiểm tra đường dẫn file
PermissionError File đang mở Đóng Microsoft Word trước khi chạy
```

**Bước cuối là chạy lại file check_fix.ipynb**
Vì sẽ có những file doc,docs,DOC bị lỗi trong quá trình convert nên phải check lại là fix thủ công nếu cần
