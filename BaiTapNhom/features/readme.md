
# Features Folder

Thư mục `features/` chứa các đặc trưng MFCC đã được trích xuất từ bộ dữ liệu FSD50K, phục vụ cho hai loại mô hình:

## 📁 Cấu trúc thư mục

```
features/
├── rf/
│   ├── X_dev.npy        # Đặc trưng đầu vào cho tập phát triển (flatten MFCC)
│   ├── y_dev.npy        # Nhãn tương ứng cho tập phát triển
│   ├── X_eval.npy       # Đặc trưng đầu vào cho tập đánh giá
│   └── y_eval.npy       # Nhãn tương ứng cho tập đánh giá
├── cnn/
    ├── X_dev.npy        # Đặc trưng dạng ma trận MFCC cho tập phát triển (shape: 100 × 13)
    ├── y_dev.npy
    ├── X_eval.npy
    └── y_eval.npy
```

## 🧪 Ghi chú:

- `rf/`: chứa đặc trưng đã được **làm phẳng** (1D) phù hợp với các mô hình truyền thống như Random Forest, SVM,...
- `cnn/`: chứa đặc trưng **giữ nguyên hình dạng 2D** (100, 13) dành cho mô hình CNN, RNN hoặc các mô hình học sâu khác.

Mỗi tệp `.npy` tương ứng với tập huấn luyện (`dev`) và tập đánh giá (`eval`), được lưu trữ sẵn để sử dụng nhanh trong huấn luyện mô hình.
