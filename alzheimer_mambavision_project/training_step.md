# Các bước tải dữ liệu và training

Hướng dẫn này dùng dataset Alzheimer MRI 4 lớp trên Kaggle. Mở [link_dataset.md](link_dataset.md), đăng nhập Kaggle và tải file ZIP trước khi bắt đầu.

## 1. Chuẩn bị môi trường

Code Mamba và CUDA extension trong dự án được cung cấp cho Linux; không chạy `install_mamba.sh` trực tiếp bằng PowerShell Windows. Trên máy Windows, dùng môi trường Linux như WSL2 Ubuntu. Các lệnh bên dưới chạy trong terminal Linux/WSL.

Mở terminal WSL, chuyển đến thư mục dự án (đường dẫn `/mnt/c/...` là ổ C: trong WSL):

```bash
cd /mnt/c/Users/ADMIN/Desktop/alzheimer_mambavision_workstation/alzheimer_mambavision_project
```

Cài Python và PyTorch/torchvision tương thích với CUDA của máy theo hướng dẫn cài đặt trong [README.md](README.md). Sau đó, từ thư mục dự án, cài các gói của dự án:

```bash
bash install_mamba.sh
python -m pip install -r requirements.txt
```

Nếu máy không có GPU NVIDIA/CUDA, hãy xem hướng dẫn trong README và kiểm tra tính tương thích của `mamba-ssm` cùng các CUDA extension; script cài đặt hiện tại nhắm tới CUDA trên Linux.

## 2. Tải và kiểm tra dataset

1. Tải file ZIP từ [trang dataset Kaggle](https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented) bằng nút **Download**.
2. Lưu file ZIP vào một thư mục dễ tìm, ví dụ `C:\Users\ADMIN\Downloads`.
3. Dự án nhận file ZIP trực tiếp; không bắt buộc phải giải nén trước. Nếu muốn giải nén, truyền thư mục chứa các ảnh thay cho đường dẫn ZIP.
4. Dữ liệu phải có các thư mục lớp `MildDemented`, `ModerateDemented`, `NonDemented` và `VeryMildDemented`. ZIP có thể chứa thêm một thư mục cha; chương trình sẽ tìm các thư mục lớp bên trong.

Trong WSL, ổ C: thường được truy cập dưới `/mnt/c`. Thay `ten-file-dataset.zip` bên dưới bằng tên ZIP thực tế.

## 3. Chạy training và đánh giá

Từ thư mục gốc dự án trong terminal WSL:

```bash
python train.py \
  --data-root "/mnt/c/Users/ADMIN/Downloads/ten-file-dataset.zip" \
  --output-dir "./mri_mambavision_output"
```

Nếu đã giải nén dữ liệu, dùng đường dẫn đến thư mục chứa các thư mục lớp:

```bash
python train.py \
  --data-root "/mnt/c/Users/ADMIN/Downloads/ten-thu-muc-dataset" \
  --output-dir "./mri_mambavision_output"
```

Có thể điều chỉnh batch size, số epoch và các tham số khác. Ví dụ:

```bash
python train.py \
  --data-root "/mnt/c/Users/ADMIN/Downloads/ten-file-dataset.zip" \
  --output-dir "./mri_mambavision_output" \
  --batch-size 8 \
  --epochs 30 \
  --learning-rate 2e-5 \
  --patience 6 \
  --num-workers 0
```

`train.py` tự giải nén ZIP nếu cần, loại ảnh trùng theo hash, tạo các tập train/validation/test xấp xỉ 70/15/15, huấn luyện mô hình, rồi đánh giá checkpoint tốt nhất trên tập test. Tỷ lệ chia có thể thay đổi do các ảnh trùng được giữ chung một nhóm.

## 4. Xem kết quả

Mặc định kết quả nằm trong `mri_mambavision_output`:

- `Alzheimer_MambaVision_Tiny_best.pth`: checkpoint có macro-F1 validation tốt nhất.
- `split_manifest.csv`: danh sách ảnh và tập dữ liệu mà ảnh được gán vào.
- Các thư mục dữ liệu đã giải nén và chia tập.
- Biểu đồ quá trình training.
- `test_confusion_matrix.png`: ma trận nhầm lẫn trên test.

Accuracy, macro-F1 và báo cáo phân loại theo lớp được in trong terminal. Evaluator được gọi tự động ở cuối `train.py`; không chạy `mri_mamba/engine/evaluator.py` trực tiếp.

## Lưu ý khi đánh giá

Dataset có thể chứa ảnh tăng cường (augmented). Dự án chỉ nhận biết các ảnh giống hệt nhau theo hash; ảnh tăng cường từ cùng một ảnh gốc có nội dung khác nhau có thể rơi vào các tập khác nhau. Vì vậy, kết quả test không nhất thiết phản ánh đánh giá tách biệt theo bệnh nhân. Nếu có mã bệnh nhân hoặc thông tin nguồn gốc ảnh, nên chia nhóm theo bệnh nhân/ảnh gốc trước khi kết luận hiệu năng.
