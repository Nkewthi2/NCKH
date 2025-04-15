
# FSD50K Dataset

[FSD50K](https://zenodo.org/record/4060432) là một bộ dữ liệu âm thanh mở lớn, được gán nhãn thủ công, dùng cho các bài toán nhận dạng và phân loại sự kiện âm thanh. Bộ dữ liệu được phát triển bởi Nhóm Công nghệ Âm nhạc tại Đại học Pompeu Fabra, Barcelona.

## 📦 Nội dung bộ dữ liệu

- **Tổng số đoạn âm thanh**: 51,197
- **Tổng thời lượng**: 108.3 giờ
- **Số lớp âm thanh**: 200 lớp (bao gồm 144 lớp lá và 56 lớp trung gian)
- **Hệ thống phân loại**: Dựa trên AudioSet Ontology
- **Định dạng tệp**: PCM mono, 16-bit, 44.1 kHz, WAV (không nén)
- **Gán nhãn**: Weak labels (gán nhãn ở cấp đoạn, có thể có nhiều nhãn trên một đoạn)
- **Nguồn dữ liệu**: Các đoạn âm thanh được thu thập từ cộng đồng trên Freesound.org

## 📂 Phân chia dữ liệu

- `dev` (development set): Dùng cho huấn luyện và kiểm thử.
- `eval` (evaluation set): Dùng để đánh giá mô hình. Các đoạn âm thanh được chọn từ người dùng khác nhau để tránh trùng lặp nguồn.

## 💡 Ứng dụng

FSD50K phù hợp với các tác vụ như:

- Nhận dạng sự kiện âm thanh (Audio Event Recognition)
- Phân loại âm thanh đa nhãn (Multi-label Classification)
- Học từ dữ liệu bán giám sát hoặc huấn luyện mô hình học sâu (Deep Learning for Audio)
- Xây dựng mô hình âm thanh tổng quát với từ vựng lớn

## 📑 Thông tin bổ sung

Bộ dữ liệu đi kèm với metadata chi tiết bao gồm:

- Nhãn âm thanh
- Độ nổi bật của âm thanh trong đoạn
- Thông tin người dùng và mô tả từ Freesound

## 🔗 Liên kết

- Trang chủ Zenodo: [https://zenodo.org/record/4060432](https://zenodo.org/record/4060432)
- Tài liệu chi tiết: [FSD50K: an Open Dataset of Human-Labeled Sound Events](https://arxiv.org/abs/2010.00475)

## 📜 Giấy phép

Bộ dữ liệu được phát hành theo giấy phép Creative Commons Attribution 4.0 International (CC BY 4.0).

---

> Nếu bạn sử dụng bộ dữ liệu này cho nghiên cứu, vui lòng trích dẫn bài báo gốc từ nhóm phát triển.
