# KẾ HOẠCH TRIỂN KHAI BTL INT3401: NHẬN DIỆN LOÀI GỖ DỰA TRÊN VÂN GỖ (BẢN V3 - ĐÃ XÁC THỰC THỰC NGHIỆM)

**Chủ đề:** *Wood Species Classification from Macroscopic Grain Images using Transfer Learning*  
**Môn học / Lớp:** INT3401 – Nhóm lớp 11 | **Tên file nộp:** `INT3401_11_NhomX.zip`  
**Cập nhật Bản V3:** Bản v3 cập nhật toàn bộ thông số thực tế sau khi tải và kiểm chứng thành công bộ dữ liệu chuẩn Zenodo 2545611 (WOOD-AUTH, Barmpoutis et al., 2018), chuyển đổi toàn bộ pipeline sang **12 loài gỗ thực tế** (thay cho giả định 7 loài cũ), thiết lập quy trình chia dữ liệu Train/Val/Test chống rò rỉ, hoàn thiện mã nguồn và dàn ý báo cáo IEEE LaTeX.

---

## 0. ĐỐI CHIẾU YÊU CẦU ĐỀ BÀI VÀ TRẠNG THÁI TRIỂN KHAI

| STT | Yêu cầu đề bài | Giải pháp kỹ thuật (Bản V3) | Trạng thái |
|:---:|---|---|:---:|
| 1 | File nộp `INT3401_11_NhomX.zip` gồm báo cáo PDF + source code + tài liệu, **không gồm dataset** | Script `scripts/make_submission_zip.py`, `.gitignore` loại trừ `dataset/`, `*.zip`, `.pth` | Sẵn sàng |
| 2 | Báo cáo LaTeX chuẩn IEEE (`IEEE-conference-template-062824.tex`), tiếng Việt, **≤ 7 trang** | Thư mục `report/main.tex` đã cấu hình `[vietnamese]{babel}`, T5 fontenc, dàn ý 7 trang | Đã lên khung |
| 3 | Bảng phân công công việc cụ thể, có tỷ lệ đóng góp | Bảng phân công chi tiết khớp với các task CoReach | Hoàn thiện |
| 4 | Phát biểu bài toán, làm rõ thách thức fine-grained | Định nghĩa input (ảnh RGB macro 200x200), output (12 loài gỗ + xác suất + Grad-CAM) | Hoàn thiện |
| 5 | Dữ liệu thực tế: kiểm chứng số lớp, phân bố | Đã xác thực Zenodo 2545611: 8,544 ảnh macro, **12 lớp**, tỷ lệ Train (5,708) / Test (2,836) | Đã xác thực |
| 6 | Baseline cổ điển so sánh với Deep Learning | `src/baseline_lbp_svm.py`: LBP (Local Binary Pattern) + GLCM + Support Vector Machine | Đang hoàn thiện |
| 7 | Transfer Learning 2 giai đoạn (ResNet18, ResNet50, ConvNeXt) | `src/train.py`: Giai đoạn 1 đóng băng backbone, Giai đoạn 2 fine-tuning đa tốc độ học | Đang hoàn thiện |
| 8 | Ablation & Robustness kiểm chứng giả thuyết màu sắc | `src/evaluate.py`: So sánh có/không Color Augmentation, 7 biến đổi màu (grayscale, hue, saturation, brightness) | Đang hoàn thiện |
| 9 | Trực quan hóa và giải thích mô hình (Grad-CAM) | `src/gradcam.py`: Tạo heatmap cho cả mẫu phân loại đúng và mẫu phân loại sai | Đang hoàn thiện |
| 10 | ≥ 7 tài liệu tham khảo chuẩn IEEE | `report/refs.bib`: 12 tài liệu trích dẫn chuẩn IEEE (Barmpoutis 2018, He 2016, Selvaraju 2017, ...) | Đã chuẩn bị |
| 11 | Mã nguồn GitHub, README hướng dẫn chạy lại từ repo sạch | `README.md` chi tiết từ setup venv, tải dữ liệu, train, evaluate, demo | Sẵn sàng |
| 12 | Thiết lập CoReach: nhóm `2627I_INT3401_11_Nhóm X`, Supervisor `duyld@vnu.edu.vn` | Hướng dẫn tạo 6 task tương ứng 6 mốc tiến độ trong kế hoạch | Sẵn sàng |

---

## 1. THÔNG TIN BỘ DỮ LIỆU THỰC NGHIỆM ĐÃ XÁC THỰC

Từ việc tải và kiểm tra trực tiếp bản ghi Zenodo 2545611:
- **Tên bộ dữ liệu:** Wood Species Dataset (WOOD-AUTH) – Aristotle University of Thessaloniki & Imperial College London.
- **Tác giả & Trích dẫn:** Panagiotis Barmpoutis, Kosmas Dimitropoulos, Ioannis Barboutis, Nikos Grammalidis, Panagiotis Lefakis (2018), *Computers and Electronics in Agriculture*, Vol. 144, pp. 241–248. DOI: `10.5281/zenodo.2545611`.
- **Tổng số ảnh:** 8,544 ảnh cận cảnh macro (kích thước gốc 200 × 200 pixel, hệ màu RGB).
- **Số loài thực tế:** **12 loài** (được đánh số nhãn từ 1 đến 12).
  - Danh mục 12 loài gỗ:
    1. *Fagus sylvatica* (Beech / Dẻ gai)
    2. *Robinia pseudoacacia* (Black Locust / Keo dậu hoa trắng)
    3. *Prunus avium* (Wild Cherry / Anh đào dại)
    4. *Castanea sativa* (Sweet Chestnut / Dẻ Trùng Khánh)
    5. *Pseudotsuga menziesii* (Douglas Fir / Thông Douglas)
    6. *Larix decidua* (European Larch / Thông rụng lá Châu Âu)
    7. *Quercus robur* (English Oak / Sồi châu Âu)
    8. *Pinus sylvestris* (Scots Pine / Thông Scots)
    9. *Platanus orientalis* (Oriental Plane / Ti trăn phương đông)
    10. *Populus tremula / Populus tremuloides* (Aspen / Bạch dương rung)
    11. *Ailanthus altissima* (Tree of Heaven / Cây thiên đường)
    12. *Juglans regia* (Persian Walnut / Óc chó phương Tây)
- **Cấu trúc phân chia:**
  - Tập `Train` gốc: 5,708 ảnh.
  - Tập `Test` gốc: 2,836 ảnh.
  - Phân chia thực nghiệm chuẩn trong đồ án:
    - **Tập Train (học mô hình):** 4,566 ảnh (80% của Train gốc, stratified theo 12 lớp).
    - **Tập Val (chọn best checkpoint & early stopping):** 1,142 ảnh (20% của Train gốc, stratified theo 12 lớp).
    - **Tập Test (kiểm thử độc lập cuối cùng):** 2,836 ảnh (100% của Test gốc).

---

## 2. KIẾN TRÚC MÃ NGUỒN DỰ ÁN

```text
wood-grain-classification/
├── configs/
│   ├── resnet18.yaml            # Cấu hình chính (mặc định)
│   ├── resnet18_no_color.yaml   # Ablation: Không color augmentation
│   ├── resnet50.yaml            # Thí nghiệm độ sâu
│   └── convnext_tiny.yaml       # Thí nghiệm kiến trúc hiện đại
├── dataset/                     # Thư mục dữ liệu (bị gitignore)
│   ├── raw/Wood Dataset/        # Dữ liệu gốc sau khi tải
│   └── splits.json              # Danh sách phân chia Train/Val/Test
├── scripts/
│   ├── download_dataset.py      # Tự động tải từ Zenodo và giải nén
│   ├── prepare_dataset.py       # Kiểm tra ảnh, chia train/val/test, xuất splits.json
│   ├── run_all.py               # Chạy toàn bộ thực nghiệm E1 -> E8 trên mọi OS
│   └── make_submission_zip.py   # Tự động đóng gói INT3401_11_NhomX.zip chuẩn quy định
├── src/
│   ├── __init__.py
│   ├── utils.py                 # Seed, logging, load/save JSON & YAML, GPU/CPU selector
│   ├── dataset.py               # WoodDataset, DataLoaders, Color Augmentation, Robustness Shifts
│   ├── model.py                 # WoodNet (ResNet18/50, ConvNeXt-T), custom head, freezing/unfreezing
│   ├── baseline_lbp_svm.py      # Baseline: Trích xuất đặc trưng LBP + GLCM, phân loại SVM
│   ├── train.py                 # Huấn luyện 2 giai đoạn (Head warm-up + Full fine-tuning)
│   ├── evaluate.py              # Đánh giá Test, Confusion Matrix, Classification Report, Robustness Test
│   ├── gradcam.py               # Tạo và trực quan hóa bản đồ kích hoạt Grad-CAM
│   └── predict.py               # Suy luận dự đoán 1 ảnh độc lập (CLI)
├── app.py                       # Giao diện web demo tương tác bằng Streamlit
├── outputs/                     # Chứa các bảng số liệu, ma trận nhầm lẫn, biểu đồ kết quả
│   ├── figures/                 # Hình ảnh Grad-CAM, Confusion Matrix, Learning curves
│   ├── metrics/                 # JSON kết quả, bảng classification report
│   └── checkpoints/             # Lưu checkpoint mô hình tốt nhất (best_model.pth)
├── report/
│   ├── main.tex                 # File nguồn báo cáo LaTeX chuẩn IEEE tiếng Việt (≤ 7 trang)
│   ├── refs.bib                 # Danh mục tài liệu tham khảo IEEE BibTeX
│   └── figures/                 # Các hình ảnh chèn vào báo cáo
├── requirements.txt             # Danh sách thư viện và phiên bản
├── .gitignore                   # Loại bỏ triệt để dữ liệu và checkpoint nặng
└── README.md                    # Hướng dẫn chi tiết cách tái lập thực nghiệm
```

---

## 3. THIẾT KẾ CÁC THỰC NGHIỆM KHOA HỌC (E1 – E8)

| Mã | Tên thí nghiệm | Phương pháp | Mục tiêu khoa học |
|:---:|---|---|---|
| **E1** | Baseline Cổ điển | LBP (P=8, R=1) + GLCM + SVM (RBF kernel) | Đặt mốc cơ sở chứng minh ưu thế của Deep Learning |
| **E2** | ResNet18 Chính | Pretrained ImageNet, 2 giai đoạn huấn luyện, Color Aug | Mô hình cốt lõi đạt cân bằng tốt giữa tốc độ và độ chính xác |
| **E3** | ResNet50 | Tăng độ sâu mô hình, 2 giai đoạn huấn luyện | Đánh giá xem mô hình sâu hơn có giúp tăng Macro-F1 trên texture không |
| **E4** | ConvNeXt-Tiny | Kiến trúc CNN hiện đại thập niên 2020 | Đánh giá tiềm năng của kiến trúc CNN tiên tiến nhất |
| **E5** | Ablation Transfer Learning | Huấn luyện từ đầu (Scratch) vs Chỉ train Head vs Fine-tune 2 giai đoạn | Chứng minh tầm quan trọng của việc fine-tune 2 giai đoạn |
| **E6** | Ablation Color Augmentation | Huấn luyện ResNet18 có vs không có Color Augmentation | Kiểm chứng giả thuyết: ép mạng học cấu trúc vân thay vì dựa vào màu |
| **E7** | Robustness Test màu sắc | Kiểm tra trên Test với 7 biến dạng: Original, Grayscale, Desaturate, Hue ±0.2, Bright, Dark | Trả lời câu hỏi phản biện: "Gỗ bị sơn màu hoặc phủ vecni thì mô hình có bị nhầm không?" |
| **E8** | Trực quan hóa Grad-CAM | Trích xuất gradient tại lớp tích chập cuối cùng (`layer4`) trên các mẫu đúng & sai | Khẳng định mô hình tập trung vào vân gỗ hay nền ảnh |

---

## 4. CHI TIẾT HUẤN LUYỆN 2 GIAI ĐOẠN (TWO-STAGE TRAINING)

### Giai đoạn 1: Warm-up Head (Đóng băng Backbone)
- **Số epoch:** 5 epoch.
- **Trạng thái backbone:** `freeze_backbone()`, đặt ở chế độ `eval()` để giữ nguyên running mean/variance của BatchNorm.
- **Bộ tối ưu:** AdamW tối ưu riêng các tham số của Classifier Head ($lr = 10^{-3}$, weight decay $= 10^{-2}$).
- **Mục tiêu:** Tránh làm phá hỏng các trọng số đặc trưng pretrained của ImageNet do gradient ngẫu nhiên lớn ban đầu từ classifier head chưa hội tụ.

### Giai đoạn 2: Fine-tuning Toàn Bộ (Mở khóa Backbone)
- **Số epoch:** 15 epoch (có Early Stopping với patience = 5 dựa trên Validation Macro-F1).
- **Trạng thái backbone:** `unfreeze_backbone()`, mở khóa gradient toàn bộ mạng.
- **Phân tách tốc độ học (Differential Learning Rates):**
  - Backbone: $lr = 10^{-4}$ (học chậm để tinh chỉnh nhẹ đặc trưng vân gỗ).
  - Head: $lr = 10^{-3}$.
- **Bộ lập lịch:** `CosineAnnealingLR` giảm dần $lr$ theo hàm cosine.
- **Hàm mất mát:** `CrossEntropyLoss(label_smoothing=0.1)` giúp giảm hiện tượng overconfidence.
- **Tiêu chí chọn mô hình:** Lưu checkpoint khi **Macro-F1 trên tập Validation** đạt giá trị cao nhất. Tuyệt đối không dùng tập Test để chọn checkpoint!

---

## 5. DÀN Ý VÀ NỘI DUNG BÁO CÁO LATEX (IEEE, TIẾNG VIỆT, ≤ 7 TRANG)

1. **Trang 1: Tiêu đề, Tóm tắt, Bảng phân công, Phần I (Giới thiệu)**
   - Tên bài báo: *Nhận diện loài gỗ từ ảnh vân gỗ cận cảnh sử dụng học chuyển giao sâu*.
   - Tóm tắt: Nêu rõ bài toán 12 loài, bộ dữ liệu WOOD-AUTH 8,544 ảnh, kết quả chính (Macro-F1, Accuracy), **đường dẫn GitHub**.
   - Bảng phân công công việc cụ thể (thành viên, nhiệm vụ, mức độ hoàn thành 100%).
   - Phần I: Động lực thực tế (chống buôn lậu gỗ quý, kiểm định nội thất), phát biểu bài toán, các thách thức đặc thù (góc cắt, sơn phủ vecni, vân tương tự nhau).
2. **Trang 2: Phần II (Công trình liên quan) & Phần III (Phương pháp - Dữ liệu & Tiền xử lý)**
   - Phân tích các phương pháp cổ điển (LBP, GLCM, SVM) và CNN (ResNet, ConvNeXt).
   - Thống kê bộ dữ liệu WOOD-AUTH 12 loài, quy trình phân chia tập dữ liệu chống rò rỉ.
   - Các phép biến đổi dữ liệu (Color Augmentation, Random Crop, Rotation).
3. **Trang 3: Phần III (Phương pháp - Kiến trúc & Huấn luyện 2 giai đoạn)**
   - Sơ đồ kiến trúc WoodNet.
   - Giải thích chiến lược huấn luyện 2 giai đoạn và cơ chế phân tách tốc độ học.
   - Cơ sở toán học của Grad-CAM.
4. **Trang 4–5: Phần IV (Thực nghiệm & Kết quả)**
   - Bảng tổng hợp kết quả chính (E1 – E4): Accuracy, Macro-F1, độ trễ suy luận CPU.
   - Bảng kết quả Ablation (E5, E6): Tác động của transfer learning và color augmentation.
   - Biểu đồ/Bảng Robustness Test (E7): Phân tích chi tiết khả năng chống chịu khi gỗ bị đổi màu.
   - Ma trận nhầm lẫn (Confusion Matrix) trên 12 loài: Phân tích các cặp loài hay nhầm lẫn (ví dụ giữa các loài cùng họ thông hoặc sồi).
5. **Trang 6: Phân tích lỗi & Trực quan hóa Grad-CAM (E8)**
   - Trình bày hình ảnh Grad-CAM trên các mẫu dự đoán đúng và mẫu dự đoán sai.
   - Thảo luận các trường hợp biên và nguyên nhân gây sai lệch.
6. **Trang 7: Phần V (Kết luận, Hạn chế & Hướng phát triển) & Tài liệu tham khảo**
   - Tóm tắt đóng góp, chỉ ra các hạn chế (ảnh chụp phòng thí nghiệm chưa phủ hết mọi góc cắt thực tế).
   - Danh mục ≥ 7 tài liệu tham khảo định dạng IEEE.

---

## 6. DANH SÁCH TÀI LIỆU THAM KHẢO CHUẨN IEEE

1. P. Barmpoutis, K. Dimitropoulos, I. Barboutis, N. Grammalidis, and P. Lefakis, "Wood species recognition through multidimensional texture analysis," *Comput. Electron. Agric.*, vol. 144, pp. 241–248, 2018.
2. P. Barmpoutis, "Wood Species Dataset," *Zenodo*, 2019. DOI: 10.5281/zenodo.2545611.
3. K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE CVPR*, 2016, pp. 770–778.
4. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in *Proc. IEEE ICCV*, 2017, pp. 618–626.
5. Z. Liu et al., "A ConvNet for the 2020s," in *Proc. IEEE CVPR*, 2022, pp. 11976–11986.
6. T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 24, no. 7, pp. 971–987, 2002.
7. R. Geirhos et al., "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness," in *Proc. ICLR*, 2019.
8. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019.
9. J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?" in *Proc. NeurIPS*, 2014, pp. 3320–3328.
10. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, 2019, pp. 8024–8035.

---

## 7. KẾ HOẠCH BƯỚC TIẾP THEO

- [x] Bước 1: Cài đặt môi trường Python 3.10, PyTorch, Scikit-learn, OpenCV, Streamlit.
- [x] Bước 2: Tải và kiểm chứng bộ dữ liệu Zenodo 2545611 (8,544 ảnh, 12 lớp).
- [ ] Bước 3: Triển khai `scripts/prepare_dataset.py` để làm sạch, kiểm tra ảnh và chia `splits.json`.
- [ ] Bước 4: Hoàn thiện `src/model.py`, `src/train.py`, `src/baseline_lbp_svm.py`, `src/evaluate.py`, `src/gradcam.py`, `src/predict.py`.
- [ ] Bước 5: Viết các file cấu hình `configs/*.yaml`.
- [ ] Bước 6: Chạy thực nghiệm hoàn chỉnh (E1 baseline, E2 ResNet18, E6 Ablation, E7 Robustness, E8 Grad-CAM) và lưu toàn bộ kết quả vào `outputs/`.
- [ ] Bước 7: Viết ứng dụng demo tương tác `app.py` bằng Streamlit.
- [ ] Bước 8: Hoàn thiện báo cáo LaTeX `report/main.tex` và `refs.bib`.
- [ ] Bước 9: Tạo script đóng gói `scripts/make_submission_zip.py` và hoàn thiện `README.md`.
