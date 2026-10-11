---
title: Wood Grain Species Classification INT3401
emoji: 🪵
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: mit
---

# 🪵 Hệ Thống Nhận Diện Loài Gỗ Cận Cảnh (Wood Grain Classification)

Showcase tương tác kết quả **Bài tập lớn môn Học Máy (INT3401) - Lớp 11**, Trường Đại học Công nghệ (VNU-UET).

## 🔬 Thông Tin Nghiên Cứu
- **Mô hình đề xuất:** ResNet-18 Deep Transfer Learning (Two-Stage Training: Head Warm-up + Fine-tuning)
- **Tăng cường dữ liệu:** Thích ứng biến đổi màu sắc (Color Augmentation) chống trượt phân phối ánh sáng
- **Giải thích quyết định:** Grad-CAM (Gradient-weighted Class Activation Mapping) trên lớp tích chập cuối `layer4`
- **Kết quả thực nghiệm:**
  - **Test Accuracy:** 92.74%
  - **Macro-F1:** 92.57%
  - **Độ trễ suy luận (CPU):** 14.73 ms / ảnh
- **Tập dữ liệu:** Zenodo DOI 10.5281/zenodo.2545611 (8,544 ảnh RGB thớ gỗ, 12 loài gỗ châu Âu)
