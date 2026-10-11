# Link dataset

Dataset Alzheimer MRI 4 lớp trên Kaggle:

<https://www.kaggle.com/datasets/aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented>

Bạn cần đăng nhập Kaggle và chấp nhận các điều khoản dataset (nếu được yêu cầu) trước khi tải.

## Tải dữ liệu

1. Mở link Kaggle ở trên và đăng nhập tài khoản.
2. Chọn **Download** trên trang dataset để tải file ZIP về máy. Tên file ZIP có thể thay đổi.
3. Giữ nguyên file ZIP hoặc giải nén. Khi chạy `train.py`, truyền đường dẫn đến file ZIP hoặc thư mục dữ liệu đã giải nén bằng `--data-root`.
4. Kiểm tra dữ liệu có đủ bốn thư mục lớp mà dự án yêu cầu:
   - `MildDemented`
   - `ModerateDemented`
   - `NonDemented`
   - `VeryMildDemented`

Xem [training_step.md](training_step.md) để biết các bước cài môi trường và chạy huấn luyện.

## Lưu ý

Dự án loại trùng bằng hash nội dung ảnh và chia nhóm ảnh trùng giữa train/validation/test. Việc này không đảm bảo các ảnh thuộc cùng một bệnh nhân được giữ chung một tập. Nếu dataset có nhiều ảnh tăng cường (augmented) từ cùng ảnh gốc nhưng không có thông tin liên kết, kết quả đánh giá có thể bị ảnh hưởng bởi rò rỉ dữ liệu.
