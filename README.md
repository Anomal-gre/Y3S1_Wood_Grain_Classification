# Phân Loại Loài Gỗ Từ Ảnh Cận Cảnh Vân Gỗ (Wood Species Texture Classification)

> **BTL Học Máy (INT3401) – Lớp 11**  
> **Nhóm thực hiện:** Nhóm 11  
> **Mã nguồn:** [GitHub Repository](https://github.com/INT3401-Group11/Wood-Grain-Classification)

---

## 📌 1. Giới Thiệu Đề Tài

Bài toán nghiên cứu: **Nhận diện loài gỗ từ ảnh cấu trúc vân cận cảnh (Macroscopic Grain Images)** bằng phương pháp Học Chuyển Giao Hai Giai Đoạn (Two-Stage Transfer Learning).

Hệ thống giải quyết 3 thách thức cốt lõi:
1. **Phân loại kết cấu tinh vi (Fine-grained Texture Classification):** Phân biệt 12 loài gỗ có hoa văn mạch gỗ tương đồng.
2. **Giảm phụ thuộc vào màu sắc bề mặt:** Mô phỏng hiện tượng gỗ bị sơn phủ, nhuộm vecni hoặc phai màu bằng kỹ thuật **Color Augmentation** và kiểm chứng độ bền vững qua **Robustness Test (E7)**.
3. **Giải thích quyết định:** Trực quan hóa vùng kích hoạt đặc trưng giải phẫu thớ gỗ bằng **Grad-CAM (E8)**.

---

## 📊 2. Bộ Dữ Liệu Thực Nghiệm (WOOD-AUTH)

- **Nguồn dữ liệu:** Aristotle University of Thessaloniki & Imperial College London (Barmpoutis et al., 2018).
- **DOI bản ghi Zenodo:** `10.5281/zenodo.2545611`.
- **Tổng số ảnh:** 8,544 ảnh RGB cận cảnh (kích thước gốc 200 × 200 pixel).
- **12 loài gỗ:**
  1. Dẻ gai (*Fagus sylvatica*)
  2. Keo dậu (*Robinia pseudoacacia*)
  3. Anh đào dại (*Prunus avium*)
  4. Dẻ Trùng Khánh (*Castanea sativa*)
  5. Thông Douglas (*Pseudotsuga menziesii*)
  6. Thông rụng lá (*Larix decidua*)
  7. Sồi châu Âu (*Quercus robur*)
  8. Thông Scots (*Pinus sylvestris*)
  9. Ti trăn (*Platanus orientalis*)
  10. Bạch dương rung (*Populus tremula*)
  11. Cây thiên đường (*Ailanthus altissima*)
  12. Óc chó phương Tây (*Juglans regia*)
- **Phân chia không rò rỉ:**
  - Tập Huấn luyện (Train): 4,566 ảnh (phân tầng).
  - Tập Thẩm định (Validation): 1,142 ảnh (dùng chọn checkpoint tốt nhất & early stopping).
  - Tập Kiểm thử độc lập (Test): 2,836 ảnh (chuẩn benchmark gốc).

---

## 🏆 3. Kết Quả Thực Nghiệm

| Mô hình | Cấu hình | Test Accuracy | Test Macro-F1 | Weighted-F1 | Độ trễ CPU (ms/ảnh) |
|---|---|:---:|:---:|:---:|:---:|
| **Baseline LBP-GLCM + SVM (E1)** | RBF Kernel (C=10, gamma=scale) | 80.99% | 78.92% | 81.02% | **0.36 ms** |
| **ResNet18 (Đề xuất - E2)** | Two-stage (15 ep), Color Aug | **92.74%** | **92.57%** | **92.75%** | 14.73 ms |
| **ResNet18 (Ablation E6)** | Two-stage, Không Color Aug | 92.80% | 92.40% | 92.70% | 14.70 ms |
| **ResNet50** | Two-stage, Color Aug | 93.41% | 93.15% | 93.40% | 32.40 ms |
| **ConvNeXt-Tiny** | Two-stage, Color Aug | 93.85% | 93.52% | 93.80% | 38.10 ms |

### Thử nghiệm độ bền vững với màu sắc (Robustness Test E7):
Đánh giá trên 7 biến dạng màu sắc mô phỏng gỗ bị sơn màu hoặc phủ vecni:
- **Ảnh gốc (Original):** `92.57%` Macro-F1
- **Ảnh xám (Grayscale):** `82.06%` Macro-F1
- **Giảm bão hòa (Desaturate):** `88.82%` Macro-F1 (chỉ giảm 3.74%)
- **Lệch màu Hue +0.2:** `78.90%` Macro-F1
- **Lệch màu Hue -0.2:** `82.96%` Macro-F1
- **Ảnh tối (Dark - Brightness 0.6):** `90.74%` Macro-F1 (chỉ giảm 1.83%)
- **Ảnh sáng (Bright - Brightness 1.4):** `91.04%` Macro-F1 (chỉ giảm 1.53%)

*Kết luận:* Color Augmentation giúp mạng nơ-ron nhận diện vân gỗ độc lập với màu sắc, duy trì Macro-F1 cao ngay cả khi gỗ bị sơn phủ vecni khác màu.

---

## 🚀 4. Hướng Dẫn Cài Đặt và Chạy

### 4.1 Khởi tạo môi trường
```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt trên Windows
.venv\Scripts\activate

# Hoặc kích hoạt trên Linux/macOS
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 4.2 Tải và chuẩn bị dữ liệu
```bash
# 1. Tải dataset từ Zenodo (khoảng 47.5 MB)
python scripts/download_dataset.py

# 2. Kiểm tra ảnh và tạo splits.json chống rò rỉ
python scripts/prepare_dataset.py
```

### 4.3 Chạy Baseline cổ điển (E1)
```bash
python src/baseline_lbp_svm.py
```

### 4.4 Huấn luyện mô hình chính (Two-Stage Transfer Learning)
```bash
# Huấn luyện ResNet18 (mặc định)
python src/train.py --config configs/resnet18.yaml

# Huấn luyện Ablation (không Color Augmentation)
python src/train.py --config configs/resnet18_no_color.yaml
```

### 4.5 Đánh giá trên tập Test & Robustness Test (E7)
```bash
python src/evaluate.py --config configs/resnet18.yaml
```

### 4.6 Xuất bộ ảnh giải thích mô hình Grad-CAM (E8)
```bash
python src/gradcam.py
```

### 4.7 Dự đoán ảnh đơn lẻ (CLI)
```bash
python src/predict.py --image "dataset/raw/Wood Dataset/Test/1000_1.jpg" --vis "outputs/figures/sample_cam.jpg"
```

### 4.8 Chạy Web Demo tương tác (Streamlit)
```bash
streamlit run app.py
```

### 4.9 Tự động hóa toàn bộ thực nghiệm (E1 -> E8)
```bash
# Windows
scripts\run_all.bat
# hoặc
python scripts/run_all.py

# Linux / macOS
bash scripts/run_all.sh
```

### 4.10 Đóng gói nộp bài (`INT3401_11_NhomX.zip`)
```bash
python scripts/make_submission_zip.py --group Nhom11
```
*Ghi chú:* Script tự động đóng gói toàn bộ mã nguồn, báo cáo LaTeX, tài liệu, và **loại bỏ hoàn toàn dataset, file `.pth` và cache** theo đúng yêu cầu đề bài.

---

## 📁 5. Cấu Trúc Thư Mục Dự Án

```text
wood-grain-classification/
├── configs/                 # File cấu hình YAML cho các mô hình
├── dataset/                 # Thư mục dữ liệu & splits.json (bị gitignore)
├── outputs/                 # Kết quả metrics, biểu đồ, hình Grad-CAM
│   ├── checkpoints/         # Lưu checkpoint mô hình tốt nhất
│   ├── figures/             # Confusion matrix, đường cong huấn luyện, Grad-CAM
│   └── metrics/             # File JSON kết quả thực nghiệm
├── report/                  # Báo cáo IEEE LaTeX tiếng Việt (≤ 7 trang)
│   ├── main.tex             # File LaTeX nguồn
│   └── refs.bib             # Trích dẫn BibTeX chuẩn IEEE
├── scripts/                 # Scripts tự động tải, chia dữ liệu, chạy all, đóng gói
├── src/                     # Mã nguồn chính (dataset, model, train, evaluate, gradcam)
├── app.py                   # Giao diện web demo Streamlit
├── requirements.txt         # Danh sách thư viện Python
├── .gitignore               # Loại trừ dataset và checkpoint nặng
└── README.md                # Tài liệu hướng dẫn sử dụng
```

---

## 📚 6. Trích Dẫn Tài Liệu Tham Khảo

1. P. Barmpoutis, K. Dimitropoulos, I. Barboutis, N. Grammalidis, and P. Lefakis, *"Wood species recognition through multidimensional texture analysis,"* Computers and Electronics in Agriculture, vol. 144, pp. 241–248, 2018.
2. P. Barmpoutis, *"Wood Species Dataset,"* Zenodo, 2019. DOI: `10.5281/zenodo.2545611`.
3. K. He, X. Zhang, S. Ren, and J. Sun, *"Deep residual learning for image recognition,"* in Proc. IEEE CVPR, 2016.
4. R. R. Selvaraju et al., *"Grad-CAM: Visual explanations from deep networks via gradient-based localization,"* in Proc. IEEE ICCV, 2017.
5. Z. Liu et al., *"A ConvNet for the 2020s,"* in Proc. IEEE CVPR, 2022.
