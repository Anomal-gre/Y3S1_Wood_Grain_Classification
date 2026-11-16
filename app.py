"""Local ResNet18 demo; uses the same preprocessing as the benchmark."""
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

from src.gradcam import GradCAM, get_target_layer, overlay_cam
from src.inference import load_predictor, predict_image
from src.species import SPECIES_MAPPING

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs/resnet18.yaml"
CHECKPOINT = ROOT / "outputs/checkpoints/resnet18_color_aug_best.pth"

st.set_page_config(page_title="Nhận diện loài gỗ | Nhóm 18", page_icon="🪵", layout="wide")
st.title("Nhận diện loài gỗ từ ảnh vân")
recognition, research = st.tabs(["Nhận diện gỗ", "Thông tin dự án"])

@st.cache_resource
def cached_predictor(checkpoint_mtime):
    return load_predictor(str(CONFIG), str(CHECKPOINT), "cpu")

with recognition:
    st.write("Tải ảnh cận cảnh bề mặt gỗ. Mô hình dự đoán trong 12 loài WOOD-AUTH.")
    uploaded = st.file_uploader("Chọn ảnh JPG, PNG hoặc WEBP", type=["jpg", "jpeg", "png", "webp"])
    show_cam = st.checkbox("Hiển thị Grad-CAM", value=False)
    if uploaded is not None:
        try:
            with Image.open(uploaded) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError):
            st.error("Không đọc được ảnh. Hãy chọn một tệp ảnh hợp lệ.")
        else:
            left, right = st.columns(2)
            left.image(image, caption="Ảnh đã tải lên", use_container_width=True)
            if not CHECKPOINT.is_file():
                right.warning("Chưa có mô hình đã huấn luyện trên máy này.")
                right.code("python src/train.py --config configs/resnet18.yaml")
            else:
                with st.spinner("Đang nhận diện..."):
                    model, cfg, device = cached_predictor(CHECKPOINT.stat().st_mtime_ns)
                    results, tensor, model_view = predict_image(model, cfg, device, image)
                right.subheader(results[0][0]["vietnamese"])
                right.caption(results[0][0]["name"])
                for species, score in results:
                    right.write(f"{species['vietnamese']}: **{score * 100:.2f}%**")
                    right.progress(score)
                right.caption("Điểm softmax của mô hình, chưa được hiệu chỉnh xác suất.")
                if show_cam:
                    cam = GradCAM(model, get_target_layer(model))
                    try:
                        heatmap = cam.generate(tensor, results[0][0]["id"] - 1)
                        st.image(overlay_cam(model_view, heatmap),
                                 caption="Grad-CAM trên đúng vùng ảnh đầu vào 200 × 200")
                        st.caption("Vùng nóng biểu thị ảnh hưởng tương đối đến lớp dự đoán; "
                                   "không phải bằng chứng về một cấu trúc giải phẫu cụ thể.")
                    finally:
                        cam.close()

with research:
    st.write("INT3401, lớp 11 — Nhóm 18. Mô hình chính: ResNet18 học chuyển giao hai giai đoạn.")
    st.write("WOOD-AUTH: 8.544 ảnh; Train 4.566 / Validation 1.142 / Test 2.836.")
    st.table({"Phương pháp": ["LBP–GLCM–SVM", "ResNet18"],
              "Accuracy (%)": [80.99, 92.74], "Macro-F1 (%)": [78.92, 92.57]})
    st.caption("Kết quả của lần chạy đã lưu; chưa có kết quả kiểm chứng cho ResNet50, "
               "ConvNeXt-Tiny và đối chứng không tăng cường màu.")
    st.write("Các loài được nhận diện:")
    st.dataframe([{"Loài": x["vietnamese"], "Tên khoa học": x["scientific"]}
                  for x in SPECIES_MAPPING.values()], hide_index=True)
