# Project Proposal

## Title: Phân loại tiếng người - Tiền đề giao tiếp với AI

---

## 1. Introduction

Giao tiếp tự nhiên giữa con người và trí tuệ nhân tạo (AI) phụ thuộc chặt chẽ vào khả năng nhận diện tiếng nói của con người [1]. Trong bối cảnh đó, phân loại tiếng người (speech vs. non-speech) đóng vai trò như một bước tiền đề quan trọng, giúp hệ thống AI nhận biết chính xác khi nào người dùng đang giao tiếp, từ đó phản hồi một cách phù hợp.

Nhiệm vụ này không chỉ cần thiết trong các trợ lý ảo, thiết bị IoT, hay hệ thống tổng đài tự động mà còn là nền tảng của các mô hình nhận dạng giọng nói tiên tiến. Tuy nhiên, việc phân biệt tiếng người giữa môi trường âm thanh phức tạp và nhiễu nền vẫn còn là một thách thức kỹ thuật lớn [2].

Dự án này hướng đến việc xây dựng một mô hình học máy sử dụng mạng nơ-ron tích chập (CNN) để phân loại tiếng người, huấn luyện trên tập dữ liệu âm thanh đa dạng như FSD50K [3]. Đây là bước đầu nhằm hiện thực hóa một hệ thống AI có khả năng lắng nghe và phản hồi thông minh trong các tương tác thực tế.

---

## 2. Problem Statement

- Các hệ thống hiện tại dễ nhầm lẫn giữa tiếng người và các âm thanh phi ngôn ngữ (như tiếng ồn, tiếng nhạc, âm thanh môi trường), làm ảnh hưởng trực tiếp đến khả năng giao tiếp chính xác và tự nhiên của hệ thống AI [2].
- Các kỹ thuật phân loại dựa trên đặc trưng âm học thủ công không đủ mạnh trong môi trường thực tế phức tạp và nhiều nhiễu và thiếu khả năng mở rộng và thích ứng với dữ liệu đa dạng [4,5].
- Cần giải quyết vấn đề cân bằng giữa độ chính xác, tốc độ xử lý và khả năng tổng quát hóa.

---

## 3. Objectives

- Phân tích và đánh giá các đặc trưng âm thanh đặc trưng cho tiếng người.
- Xây dựng và huấn luyện mô hình phân loại tiếng người. Tìm kiếm mô hình tối ưu.
- Đánh giá hiệu năng mô hình trên dữ liệu thực tế và môi trường có nhiễu.
- Tối ưu hóa mô hình về mặt hiệu suất và khả năng triển khai thực tiễn.

---

## 4. Methodology

- **Dữ liệu:** Sử dụng nguồn dữ liệu âm thanh mở được chia sẻ công khai.
- **Phương pháp nghiên cứu:**
  - Áp dụng các thuật toán AI như Random Forest, XGBoost, Deep Learning để tìm kiếm mô hình tối ưu.
  - Sử dụng các kỹ thuật như Class Weights, Oversampling,... để xử lý dữ liệu.
  - So sánh hiệu suất giữa các mô hình.
- **Công cụ:** Python, Scikit-learn, TensorFlow/PyTorch.

---

## 5. Expected Outcomes

- Mô hình phân loại tiếng người hiệu quả.
- Khả năng thích ứng tốt với âm thanh thực tế.
- Bộ đặc trưng và quy trình huấn luyện tái sử dụng được.
- Tiền đề cho tích hợp hệ thống AI giao tiếp tự nhiên.

---

## 6. Planning

| Giai đoạn | Nội dung                              | Thời gian     |
|----------|----------------------------------------|---------------|
| 1        | Chuẩn bị và khảo sát                   | 1-2 tháng     |
| 2        | Thu thập và tiền xử lý dữ liệu         | 1 tháng       |
| 3        | Xây dựng và huấn luyện mô hình AI      | 2 tháng       |
| 4        | Đánh giá và tối ưu                     | 2 tháng       |
| 5        | Phân tích kết quả và viết báo cáo      | 1-2 tháng     |

---

## 7. Resources & Budget

- **Nhân lực:** 1 sinh viên tổng hợp và xử lý dữ liệu, 1 sinh viên thiết kế thuật toán, 1 sinh viên tổng hợp báo cáo.
- **Công cụ:** GPU / Cloud computing.

---

## 8. Conclusion

Trong bối cảnh AI ngày càng tiến gần hơn tới khả năng tương tác tự nhiên với con người, việc xây dựng một hệ thống có thể phân biệt chính xác tiếng người với các loại âm thanh phi ngôn ngữ là một bước khởi đầu không thể thiếu.

Dự án này tập trung vào việc phát triển một mô hình học sâu dựa trên CNN để thực hiện nhiệm vụ đó, sử dụng tập dữ liệu quy mô lớn và đa dạng như FSD50K. Bằng cách kết hợp giữa kỹ thuật xử lý tín hiệu âm thanh hiện đại và sức mạnh của học sâu, dự án không chỉ kỳ vọng tạo ra một mô hình phân loại hiệu quả mà còn góp phần đặt nền móng cho các hệ thống giao tiếp người – máy thông minh, linh hoạt và dễ triển khai trong thực tế.

---

## References

[1] Tavus. (n.d.). *Voice Activity Detection: What it is & How to Use it in Your Applications*. Retrieved from https://www.tavus.io/post/voice-activity-detection

[2] Way With Words. *Challenges in Speech Data Processing*. Accessed April 11, 2025.

[3] Fonseca, E., Favory, X., Pons, J., Font, F., & Serra, X. (2020). *FSD50K: An Open Dataset of Human-Labeled Sound Events*. arXiv preprint arXiv:2010.00475. https://arxiv.org/abs/2010.00475

[4] Salamon, J., & Bello, J. P. (2017). *Deep convolutional neural networks and data augmentation for environmental sound classification*. IEEE Signal Processing Letters, 24(3), 279–283. https://doi.org/10.1109/LSP.2017.2657381

[5] Thái, T. T. (2021). *Nhận dạng tiếng nói điều khiển với convolutional neural network (CNN)*. Tạp chí Khoa học Đại học Cần Thơ, 57(4), 30–39. https://doi.org/10.22144/ctu.jvn.2021.111
