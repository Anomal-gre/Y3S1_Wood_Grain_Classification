# Phân loại loài gỗ từ ảnh vân gỗ

Bài tập lớn **Học máy — INT3401, lớp 11, Nhóm 18**, Trường Đại học Công nghệ, ĐHQGHN.

Phân loại ảnh cận cảnh bề mặt gỗ thành **12 loài** trên WOOD-AUTH. Phương pháp cơ sở là **LBP + GLCM + SVM**; mô hình chính là **ResNet18 học chuyển giao hai giai đoạn**.

[Báo cáo PDF](report/main.pdf) · [Nguồn LaTeX](report/main.tex) · [Demo HF tùy chọn](https://huggingface.co/spaces/AnomalGreatad/wood-grain-classification) · [Dataset Zenodo](https://zenodo.org/records/2545611)

## Thành viên

| Thành viên | Mã sinh viên |
|---|---|
| Trần Gia Bảo | 24022949 |
| Nguyễn Đình Hùng | 24023001 |
| Trần Quang Khải | 24023009 |
| Vũ Tuấn Vương | 24023097 |

## Kết quả đã ghi nhận

Train **4.566**, Validation **1.142**, Test **2.836** ảnh.

| Phương pháp | Accuracy | Macro-F1 |
|---|---:|---:|
| LBP + GLCM + SVM (RBF) | 80,99% | 78,92% |
| ResNet18 + tăng cường màu | **92,74%** | **92,57%** |

ResNet18 cải thiện **13,65 điểm phần trăm Macro-F1**. [Kết quả gốc](outputs/metrics) thuộc một lần chạy, chưa phải trung bình nhiều seed. ResNet50, ConvNeXt-Tiny và đối chứng không tăng cường màu **chưa có số liệu kiểm chứng**; các cấu hình được giữ để thực nghiệm tiếp.

Mô hình còn phụ thuộc màu: Macro-F1 giảm còn 82,06% với ảnh xám và 78,90% khi lệch sắc độ +0,2. Chưa kiểm chứng trên gỗ nhuộm/phủ vecni thực tế.

Thời gian baseline 0,36 ms/ảnh chỉ tính StandardScaler + SVM trên vector đã có; ResNet18 14,73 ms/ảnh chỉ tính forward theo batch 64. Hai số này không phải độ trễ toàn pipeline.

## Cài đặt

Đã kiểm tra trên **Windows, Python 3.12, PyTorch 2.14.1 CPU**. Chạy các lệnh từ thư mục gốc:

~~~bash
git clone --depth 1 https://github.com/Anomal-gre/Y3S1_Wood_Grain_Classification.git
cd Y3S1_Wood_Grain_Classification
python -m venv .venv
~~~

Kích hoạt môi trường: Windows PowerShell dùng `.\.venv\Scripts\Activate.ps1`; Linux dùng `source .venv/bin/activate`. Sau đó:

~~~bash
python -m pip install -r requirements.txt
~~~

Requirements dùng wheel CPU Windows/Linux từ kho chính thức PyTorch. Nếu không kích hoạt môi trường, trên Windows thay `python` bằng `.\.venv\Scripts\python.exe`.

## Chuẩn bị dữ liệu

Dataset và trọng số không nằm trong GitHub hoặc ZIP nộp bài:

~~~bash
python scripts/download_dataset.py
python scripts/prepare_dataset.py
~~~

Ảnh nằm ở `dataset/raw/Wood Dataset/`; danh sách chia tập nằm ở `dataset/splits.json`. [split_manifest.json](configs/split_manifest.json) chỉ chứa tên file, giữ thứ tự và cách chia tập đã dùng trong báo cáo, không chứa ảnh. Seed mặc định là 42. Test không dùng chọn checkpoint; chia theo ảnh chưa bảo đảm độc lập theo mẫu gỗ vật lý.

## Huấn luyện và đánh giá

~~~bash
# Baseline: LBP 54 chiều + GLCM 10 chiều, StandardScaler, SVM RBF.
python src/baseline_lbp_svm.py

# Mô hình chính.
python src/train.py --config configs/resnet18.yaml --device cpu
python src/evaluate.py --config configs/resnet18.yaml --device cpu
python src/gradcam.py --config configs/resnet18.yaml
~~~

Lần huấn luyện đầu cần mạng để tải trọng số ImageNet. Mô hình được lưu tại `outputs/checkpoints/baseline_lbp_glcm_svm.joblib` và `outputs/checkpoints/resnet18_color_aug_best.pth`.

| Thiết lập ResNet18 | Giá trị |
|---|---|
| Đầu vào / batch | 200 × 200 / 64 |
| Huấn luyện head | 5 epoch, learning rate 0,001 |
| Tinh chỉnh toàn mạng | 10 epoch; backbone 0,0001, head 0,001 |
| Tối ưu | AdamW, weight decay 0,01; CosineAnnealingLR ở giai đoạn 2 |
| Regularization | Dropout 0,4 / 0,3; label smoothing 0,1 |
| Chọn checkpoint | Macro-F1 Validation cao nhất |

Đổi cấu hình tại `configs/resnet18.yaml`. Chưa có tìm kiếm siêu tham số chứng minh cấu hình là tối ưu.

~~~bash
# Toàn bộ quy trình chính: huấn luyện lại và ghi lại kết quả trong outputs/.
python scripts/run_all.py

# Thêm đối chứng không tăng cường màu (tùy chọn, cần thêm thời gian).
python scripts/run_all.py --ablation
~~~

Không cần chạy lại toàn bộ huấn luyện để đọc báo cáo hoặc kiểm tra mã.

## Nhận diện và demo local

Cần checkpoint đã huấn luyện. Chương trình báo thiếu mô hình thay vì dự đoán bằng trọng số ngẫu nhiên:

~~~bash
python src/predict.py --image "duong/dan/anh.jpg" --vis "outputs/figures/prediction_cam.jpg"
python -m streamlit run app.py
~~~

Demo local có tab tải ảnh nhận diện và tab thông tin dự án, Top-3 và Grad-CAM tùy chọn. CLI và demo dùng chung tiền xử lý 200 × 200 với benchmark. Heatmap phủ lên đúng vùng ảnh sau resize/crop; điểm softmax chưa được hiệu chỉnh xác suất.

**Hugging Face không bắt buộc** để tải dữ liệu, huấn luyện, đánh giá hoặc chạy local. Nó phục vụ minh họa trực tuyến; mã triển khai riêng ở [repository của Space](https://huggingface.co/spaces/AnomalGreatad/wood-grain-classification/tree/main). GitHub này không sao chép thêm bộ `src/`, thư viện cài sẵn hay ONNX của Space. Không coi điểm benchmark là kết quả đã đo của bản web dùng tiền xử lý khác. Cần kiểm tra trạng thái và nhận diện đầu cuối trên dịch vụ tại thời điểm trình bày.

## Kiểm tra và biên dịch báo cáo

~~~bash
python -m unittest discover -s tests -v
~~~

Báo cáo dùng IEEEtran, tiếng Việt T5/vntex. Trong thư mục `report/`:

~~~bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
~~~

Hoặc chạy `pdflatex main.tex`, `bibtex main`, rồi `pdflatex main.tex` hai lần. Giữ thư mục `figures/` đi kèm. Cần đối chiếu template gốc của giảng viên nếu yêu cầu file `IEEE-conference-template-062824.tex`; đổi tên file không chứng minh đã dùng template gốc.

## Đóng gói nộp bài

Sau khi biên dịch PDF mới:

~~~bash
python scripts/make_submission_zip.py --group 18
~~~

Tạo `dist/INT3401_11_Nhom18.zip` gồm mã nguồn, README, PDF/LaTeX và hình. Script dùng danh sách thư mục cho phép, loại dataset, checkpoint, `.venv`, `tmp/`, bản triển khai HF và ZIP cũ.

## Cấu trúc

~~~text
app.py                 Demo Streamlit local
configs/               Cấu hình và bản kê chia tập
src/                   Dữ liệu, mô hình, huấn luyện, đánh giá, suy luận, Grad-CAM
scripts/               Tải dữ liệu, chia tập, chạy pipeline, đóng gói
tests/                 Kiểm tra suy luận, Grad-CAM và gói nộp
outputs/metrics/       Kết quả thực nghiệm đã lưu
report/                PDF, LaTeX, BibTeX và hình của báo cáo
~~~

Các thư mục `dataset/`, `outputs/checkpoints/`, `outputs/figures/`, `.venv/`, `tmp/` và `dist/` được tạo trên máy và bỏ qua bởi Git. Các bản HF cũ trên máy cũng được ignore; Space trực tuyến không bị thay đổi khi dọn repository.

## Tài liệu chính

- Barmpoutis và cộng sự, *Wood species recognition through multidimensional texture analysis*, Computers and Electronics in Agriculture, 2018.
- Barmpoutis, *Wood Species Dataset*, Zenodo, 2019. DOI: [10.5281/zenodo.2545611](https://doi.org/10.5281/zenodo.2545611).
- He và cộng sự, *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Selvaraju và cộng sự, *Grad-CAM*, ICCV 2017.

Trích dẫn đầy đủ theo IEEE nằm trong báo cáo.
