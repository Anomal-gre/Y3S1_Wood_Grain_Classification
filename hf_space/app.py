"""
Interactive Web Application for Wood Grain Species Classification.
Built with Streamlit and PyTorch with real-time Grad-CAM interpretability.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, get_device
from src.dataset import build_eval_transform
from src.model import WoodNet
from src.gradcam import GradCAM, overlay_cam, get_target_layer

# Extended botanical and wood craftsmanship knowledge base
WOOD_DETAILS = {
    "1": {
        "scientific": "Fagus sylvatica",
        "common_en": "European Beech",
        "common_vi": "Gỗ Dẻ Gai Châu Âu",
        "type": "Gỗ cứng (Hardwood)",
        "grain": "Thớ mịn, thẳng, tia gỗ ngắn dạng đốm nâu li ti phân bố đều.",
        "usage": "Nội thất cao cấp, sàn gỗ, đồ chơi mộc, cán công cụ."
    },
    "2": {
        "scientific": "Robinia pseudoacacia",
        "common_en": "Black Locust",
        "common_vi": "Gỗ Keo Dậu Hoa Trắng (Robinia)",
        "type": "Gỗ cứng (Hardwood, rất bền)",
        "grain": "Vòng năm rõ rệt, mạch gỗ xếp dạng vòng, vân nổi gợn sóng sắc nét.",
        "usage": "Trụ hàng rào ngoài trời, tà vẹt đường sắt, đóng thuyền, sàn chịu lực."
    },
    "3": {
        "scientific": "Prunus avium",
        "common_en": "Wild Cherry",
        "common_vi": "Gỗ Anh Đào Dại (Cherry)",
        "type": "Gỗ cứng (Hardwood)",
        "grain": "Vân mịn màng, sóng nhẹ lượn tự nhiên, ánh ấm sang trọng.",
        "usage": "Tủ kệ hoàng gia, nhạc cụ (đàn piano, violin), chạm khắc mỹ nghệ."
    },
    "4": {
        "scientific": "Castanea sativa",
        "common_en": "Sweet Chestnut",
        "common_vi": "Gỗ Dẻ Trùng Khánh (Castanea)",
        "type": "Gỗ cứng (Hardwood)",
        "grain": "Mạch gỗ thô dạng vòng tương tự sồi nhưng không có tia gỗ nổi hạt.",
        "usage": "Thùng ủ rượu vang, đồ gỗ ngoài trời, khung mái nhà cổ."
    },
    "5": {
        "scientific": "Pseudotsuga menziesii",
        "common_en": "Douglas Fir",
        "common_vi": "Gỗ Thông Douglas",
        "type": "Gỗ mềm (Softwood lá kim)",
        "grain": "Vòng sinh trưởng mùa xuân/thu tương phản cực mạnh, vân thẳng đều đặn.",
        "usage": "Kết cấu xây dựng chịu lực, xà gồ, ván ép chất lượng cao."
    },
    "6": {
        "scientific": "Larix decidua",
        "common_en": "European Larch",
        "common_vi": "Gỗ Thông Rụng Lá Châu Âu",
        "type": "Gỗ mềm (Softwood chịu nước)",
        "grain": "Vân dạng dải hẹp, màu nâu đỏ đậm dần về tâm, giàu tinh dầu nhựa.",
        "usage": "Ốp tường ngoại thất chịu mưa nắng, du thuyền, ván sàn ban công."
    },
    "7": {
        "scientific": "Quercus robur",
        "common_en": "English Oak",
        "common_vi": "Gỗ Sồi Châu Âu (Sồi Trắng)",
        "type": "Gỗ cứng (Hardwood danh tiếng)",
        "grain": "Tia gỗ lớn đặc trưng dạng vệt bạc (medullary rays), tom gỗ thô sang trọng.",
        "usage": "Nội thất cổ điển, sàn sồi, thùng ủ whisky và cognac."
    },
    "8": {
        "scientific": "Pinus sylvestris",
        "common_en": "Scots Pine",
        "common_vi": "Gỗ Thông Scots",
        "type": "Gỗ mềm (Softwood)",
        "grain": "Vân xoắn nhẹ, có mắt gỗ tròn tự nhiên, gỗ mùa muộn viền sẫm rõ ràng.",
        "usage": "Đồ nội thất gia đình phong cách Scandinavian, thùng bao bì mộc."
    },
    "9": {
        "scientific": "Platanus orientalis",
        "common_en": "Oriental Plane",
        "common_vi": "Gỗ Ti Trăn Phương Đông",
        "type": "Gỗ cứng (Hardwood)",
        "grain": "Vân đốm vảy báo (lacewood / ray flecks) lấp lánh khi xẻ tiếp tuyến.",
        "usage": "Veneer trang trí cao cấp, báng súng, nhạc cụ, đồ thủ công nghệ thuật."
    },
    "10": {
        "scientific": "Populus tremula",
        "common_en": "Aspen / Poplar",
        "common_vi": "Gỗ Bạch Dương Rung (Aspen)",
        "type": "Gỗ cứng mềm (Soft Hardwood)",
        "grain": "Vân mờ nhạt, đồng nhất, bề mặt thớ mịn xốp, không mùi nhựa.",
        "usage": "Phòng xông hơi Sauna (không dẫn nhiệt), que diêm, đũa ăn, bột giấy."
    },
    "11": {
        "scientific": "Ailanthus altissima",
        "common_en": "Tree of Heaven",
        "common_vi": "Gỗ Cây Thiên Đường (Ailanthus)",
        "type": "Gỗ cứng (Hardwood)",
        "grain": "Vân thô, lỗ mạch lớn tương tự tần bì (ash), bề mặt gân guốc.",
        "usage": "Đồ mộc gia dụng, đồ gỗ trang trí rustic, pallet chịu tải."
    },
    "12": {
        "scientific": "Juglans regia",
        "common_en": "Persian Walnut",
        "common_vi": "Gỗ Óc Chó Phương Tây (Walnut)",
        "type": "Gỗ cứng (Hardwood quý tộc)",
        "grain": "Vân cuộn sóng, xoáy ảo diệu, chuyển sắc socola từ nâu nhạt đến sẫm.",
        "usage": "Nội thất siêu xe, ốp đàn guitar hi-end, bàn hội nghị nguyên tấm."
    }
}

st.set_page_config(
    page_title="Wood Grain Classifier - INT3401",
    page_icon="🪵",
    layout="wide"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 5px;
    }
    .sub-header {
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 25px;
        font-size: 1.1rem;
    }
    .badge-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #2980b9;
    }
    .metric-box {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_cached_model():
    config_path = PROJECT_ROOT / "configs" / "resnet18.yaml"
    cfg = load_config(str(config_path))
    ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "resnet18_color_aug_best.pth"
    device = get_device("auto")

    model = WoodNet(
        backbone="resnet18",
        num_classes=12,
        pretrained=False
    ).to(device)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        status_msg = f"Đã tải checkpoint tối ưu (Epoch {ckpt.get('epoch', '?')}, Val Macro-F1: {ckpt.get('val_f1', 0)*100:.1f}%)"
    else:
        status_msg = "[Chế độ demo] Đang chạy với trọng số khởi tạo (chưa train xong checkpoint)"
    
    eval_tf = build_eval_transform(img_size=224)
    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    return model, eval_tf, gradcam, device, status_msg

st.markdown("<h1 class='main-header'>🪵 HỆ THỐNG NHẬN DIỆN LOÀI GỖ TỪ VÂN GỖ CẬN CẢNH</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>BTL Học Máy (INT3401) – Lớp 11 | Mô hình Deep Transfer Learning (ResNet18 / Two-Stage Training)</p>", unsafe_allow_html=True)

model, eval_tf, gradcam, device, model_status = load_cached_model()
st.sidebar.info(f"**Trạng thái thiết bị:** {device}\n\n{model_status}")

st.sidebar.markdown("### ⚙️ Tùy chọn nguồn ảnh")
input_mode = st.sidebar.radio("Chọn nguồn dữ liệu:", ["Ảnh mẫu trong tập kiểm thử (Test)", "Tải ảnh từ máy tính"])

selected_img = None
selected_gt_name = None

test_dir = PROJECT_ROOT / "dataset" / "raw" / "Wood Dataset" / "Test"

if input_mode == "Ảnh mẫu trong tập kiểm thử (Test)":
    sample_options = {
        "Dẻ gai (Beech - Mẫu 1000)": ("1000_1.jpg", "1"),
        "Keo dậu (Black Locust - Mẫu 1024)": ("1024_2.jpg", "2"),
        "Anh đào dại (Wild Cherry - Mẫu 1032)": ("1032_3.jpg", "3"),
        "Dẻ Trùng Khánh (Chestnut - Mẫu 1008)": ("1008_4.jpg", "4"),
        "Thông Douglas (Douglas Fir - Mẫu 1016)": ("1016_5.jpg", "5"),
        "Thông rụng lá (European Larch - Mẫu 1004)": ("1004_6.jpg", "6"),
        "Sồi châu Âu (English Oak - Mẫu 1012)": ("1012_7.jpg", "7"),
        "Thông Scots (Scots Pine - Mẫu 1020)": ("1020_8.jpg", "8"),
        "Ti trăn (Oriental Plane - Mẫu 1028)": ("1028_9.jpg", "9"),
        "Bạch dương (Aspen - Mẫu 1036)": ("1036_10.jpg", "10"),
        "Cây thiên đường (Tree of Heaven - Mẫu 1040)": ("1040_11.jpg", "11"),
        "Óc chó (Walnut - Mẫu 1044)": ("1044_12.jpg", "12")
    }
    choice = st.sidebar.selectbox("Chọn mẫu gỗ kiểm thử:", list(sample_options.keys()))
    file_name, cls_key = sample_options[choice]
    sample_file_path = PROJECT_ROOT / "samples" / f"sample_{cls_key}.jpg"
    if not sample_file_path.exists():
        sample_file_path = test_dir / file_name

    if sample_file_path.exists():
        selected_img = Image.open(sample_file_path).convert("RGB")
        selected_gt_name = WOOD_DETAILS[cls_key]["common_vi"]
    else:
        st.warning(f"Chưa tìm thấy file {sample_file_path}. Vui lòng kiểm tra thư mục samples/ hoặc chạy scripts/download_dataset.py.")
else:
    uploaded = st.sidebar.file_uploader("Tải lên ảnh cận cảnh vân gỗ (JPG, PNG):", type=["jpg", "jpeg", "png"])
    if uploaded:
        selected_img = Image.open(uploaded).convert("RGB")

show_gradcam = st.sidebar.checkbox("Bật trực quan hóa Grad-CAM", value=True)
cam_alpha = st.sidebar.slider("Độ trong suốt Heatmap:", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

if selected_img is not None:
    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.subheader("📸 Ảnh Vân Gỗ Đầu Vào")
        st.image(selected_img, caption="Ảnh macro đầu vào (kích thước chuẩn hóa 224x224)", use_container_width=True)
        if selected_gt_name:
            st.success(f"**Nhãn thực tế (Ground Truth):** {selected_gt_name}")

    with col2:
        st.subheader("🎯 Kết Quả Nhận Diện Loài Gỗ")
        
        # Inference
        tensor = eval_tf(selected_img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

        top_indices = np.argsort(probs)[::-1][:3]
        top1_key = str(top_indices[0] + 1)
        top1_info = WOOD_DETAILS[top1_key]
        top1_prob = probs[top_indices[0]] * 100

        st.markdown(f"""
        <div class="badge-card">
            <h3 style="margin-top:0; color:#1a365d;">🏆 Dự đoán số 1: {top1_info['common_vi']}</h3>
            <p><b>Tên khoa học:</b> <i>{top1_info['scientific']}</i> ({top1_info['common_en']})</p>
            <p><b>Độ tin cậy:</b> <span style="font-size:1.3rem; font-weight:bold; color:#27ae60;">{top1_prob:.2f}%</span></p>
            <p><b>Phân nhóm:</b> {top1_info['type']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Top 3 Xác Suất Cao Nhất:")
        for rank, idx in enumerate(top_indices, 1):
            key = str(idx + 1)
            info = WOOD_DETAILS[key]
            prob = probs[idx] * 100
            st.write(f"**{rank}. {info['common_vi']}** (*{info['scientific']}*)")
            st.progress(float(prob) / 100.0)
            st.caption(f"Xác suất: {prob:.2f}%")

    st.markdown("---")
    
    col_desc, col_cam = st.columns([1.2, 1.8])
    with col_desc:
        st.subheader("📖 Đặc Tính & Ứng Dụng Loài Gỗ")
        st.markdown(f"**Cấu trúc vân nhận diện:** {top1_info['grain']}")
        st.markdown(f"**Ứng dụng chế tác chính:** {top1_info['usage']}")

    with col_cam:
        if show_gradcam:
            st.subheader("🔥 Bản Đồ Giải Thích Grad-CAM (Lớp layer4)")
            cam_map = gradcam.generate(tensor, target_class=int(top_indices[0]))
            overlay = overlay_cam(selected_img, cam_map, alpha=cam_alpha)
            
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                st.image(cam_map, caption="Heatmap kích hoạt vân gỗ", use_container_width=True, clamp=True)
            with sub_c2:
                st.image(overlay, caption="Bản đồ nhiệt phủ lên thớ gỗ", use_container_width=True)
            st.caption("ℹ️ Vùng màu đỏ/vàng thể hiện khu vực cấu trúc vân mà mạng nơ-ron tập trung cao độ để đưa ra quyết định phân loại.")
else:
    st.info("👈 Hãy chọn một ảnh mẫu từ thanh bên trái hoặc tải ảnh lên để bắt đầu nhận diện!")
