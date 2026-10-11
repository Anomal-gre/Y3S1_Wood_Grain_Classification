---
title: Wood Grain Species Classifier INT3401
emoji: 🪵
colorFrom: yellow
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
---

# 🪵 Hệ Thống Nhận Diện Loài Gỗ Cận Cảnh (Wood Grain Classification)

Ứng dụng Web tương tác phục vụ **Bài tập lớn môn Học Máy (INT3401) - Lớp 11**, Trường Đại học Công nghệ (VNU-UET).

## 📌 Giới Thiệu
Hệ thống sử dụng kỹ thuật **Deep Transfer Learning (ResNet-18)** với quy trình huấn luyện hai giai đoạn (**Two-Stage Training** gồm Head Warm-up và Fine-tuning) cùng kỹ thuật tăng cường màu sắc thích ứng (**Color Augmentation**) để phân loại 12 loài gỗ châu Âu từ ảnh cận cảnh thớ gỗ (*microscopic wood grain texture*).

## ✨ Tính Năng Nổi Bật
- **Phân loại 12 loài gỗ tự nhiên** với độ chính xác cao (**Macro-F1: 92.57%**, **Accuracy: 92.74%** trên tập Test độc lập).
- **Trực quan hóa vùng quyết định bằng Grad-CAM (Gradient-weighted Class Activation Mapping)**: Làm rõ vùng thớ gỗ, tia gỗ, lỗ mạch mà mô hình tập trung quan sát.
- **Tích hợp sẵn 12 ảnh mẫu đại diện** cho 12 loài gỗ để trải nghiệm tức thì.
- **Cơ sở tri thức thực vật học & công dụng chế tác mộc** cho từng loài gỗ.
- **Hỗ trợ tải ảnh cận cảnh tùy biến từ máy tính**.

## 📊 Thông Tin Mô Hình
- **Kiến trúc:** ResNet-18 (Pretrained ImageNet-1K)
- **Tập dữ liệu:** Zenodo DOI 10.5281/zenodo.2545611 (8,544 ảnh RGB $200 \times 200$, 12 loài)
- **Tốc độ suy luận (CPU):** ~14.7 ms / ảnh
- **Nhóm tác giả:** Nhóm 11 - INT3401 11
