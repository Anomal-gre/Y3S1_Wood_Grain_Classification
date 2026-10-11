"""
Interactive Gradio Web Application for Wood Grain Species Classification.
Features top-3 confidence prediction, Grad-CAM interpretability, and botanical details.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, get_device
from src.dataset import build_eval_transform
from src.model import WoodNet
from src.gradcam import GradCAM, overlay_cam, get_target_layer

# Species details dictionary
WOOD_DETAILS = {
    "1": {"scientific": "Fagus sylvatica", "common_en": "European Beech", "common_vi": "Gỗ Dẻ Gai Châu Âu", "type": "Gỗ cứng (Hardwood)", "grain": "Thớ mịn, thẳng, tia gỗ ngắn dạng đốm nâu li ti.", "usage": "Nội thất cao cấp, sàn gỗ, cán công cụ."},
    "2": {"scientific": "Robinia pseudoacacia", "common_en": "Black Locust", "common_vi": "Gỗ Keo Dậu Hoa Trắng (Robinia)", "type": "Gỗ cứng (Hardwood rất bền)", "grain": "Vòng năm rõ rệt, mạch gỗ xếp dạng vòng, vân nổi gợn sóng sắc nét.", "usage": "Trụ hàng rào ngoài trời, tà vẹt đường sắt, sàn chịu lực."},
    "3": {"scientific": "Prunus avium", "common_en": "Wild Cherry", "common_vi": "Gỗ Anh Đào Dại (Cherry)", "type": "Gỗ cứng (Hardwood)", "grain": "Vân mịn màng, sóng nhẹ lượn tự nhiên, ánh ấm sang trọng.", "usage": "Tủ kệ hoàng gia, nhạc cụ (đàn piano, violin), mỹ nghệ."},
    "4": {"scientific": "Castanea sativa", "common_en": "Sweet Chestnut", "common_vi": "Gỗ Dẻ Trùng Khánh (Castanea)", "type": "Gỗ cứng (Hardwood)", "grain": "Mạch gỗ thô dạng vòng tương tự sồi nhưng không có tia gỗ nổi hạt.", "usage": "Thùng ủ rượu vang, đồ gỗ ngoài trời, khung mái nhà cổ."},
    "5": {"scientific": "Pseudotsuga menziesii", "common_en": "Douglas Fir", "common_vi": "Gỗ Thông Douglas", "type": "Gỗ mềm (Softwood)", "grain": "Vòng sinh trưởng mùa xuân/thu tương phản cực mạnh, vân thẳng.", "usage": "Kết cấu xây dựng chịu lực, xà gồ, ván ép chất lượng cao."},
    "6": {"scientific": "Larix decidua", "common_en": "European Larch", "common_vi": "Gỗ Thông Rụng Lá Châu Âu", "type": "Gỗ mềm (Softwood chịu nước)", "grain": "Vân dạng dải hẹp, màu nâu đỏ đậm dần về tâm, giàu tinh dầu.", "usage": "Ốp tường ngoại thất chịu mưa nắng, du thuyền, ban công."},
    "7": {"scientific": "Quercus robur", "common_en": "English Oak", "common_vi": "Gỗ Sồi Châu Âu (Sồi Trắng)", "type": "Gỗ cứng (Hardwood danh tiếng)", "grain": "Tia gỗ lớn đặc trưng dạng vệt bạc (medullary rays), tom gỗ thô.", "usage": "Nội thất cổ điển, sàn sồi, thùng ủ whisky và cognac."},
    "8": {"scientific": "Pinus sylvestris", "common_en": "Scots Pine", "common_vi": "Gỗ Thông Scots", "type": "Gỗ mềm (Softwood)", "grain": "Vân xoắn nhẹ, có mắt gỗ tròn tự nhiên, gỗ mùa muộn viền sẫm.", "usage": "Đồ nội thất phong cách Scandinavian, thùng bao bì mộc."},
    "9": {"scientific": "Platanus orientalis", "common_en": "Oriental Plane", "common_vi": "Gỗ Ti Trăn Phương Đông", "type": "Gỗ cứng (Hardwood)", "grain": "Vân đốm vảy báo (lacewood / ray flecks) lấp lánh khi xẻ tiếp tuyến.", "usage": "Veneer trang trí cao cấp, báng súng, nhạc cụ, nghệ thuật."},
    "10": {"scientific": "Populus tremula", "common_en": "Aspen / Poplar", "common_vi": "Gỗ Bạch Dương Rung (Aspen)", "type": "Gỗ cứng mềm (Soft Hardwood)", "grain": "Vân mờ nhạt, đồng nhất, bề mặt thớ mịn xốp, không mùi nhựa.", "usage": "Phòng xông hơi Sauna (không dẫn nhiệt), que diêm, đũa ăn, bột giấy."},
    "11": {"scientific": "Ailanthus altissima", "common_en": "Tree of Heaven", "common_vi": "Gỗ Cây Thiên Đường (Ailanthus)", "type": "Gỗ cứng (Hardwood)", "grain": "Vân thô, lỗ mạch lớn tương tự tần bì (ash), bề mặt gân guốc.", "usage": "Đồ mộc gia dụng, đồ gỗ trang trí rustic, pallet chịu tải."},
    "12": {"scientific": "Juglans regia", "common_en": "Persian Walnut", "common_vi": "Gỗ Óc Chó Phương Tây (Walnut)", "type": "Gỗ cứng (Hardwood quý tộc)", "grain": "Vân cuộn sóng, xoáy ảo diệu, chuyển sắc socola từ nâu nhạt đến sẫm.", "usage": "Nội thất siêu xe, ốp đàn guitar hi-end, bàn hội nghị nguyên tấm."}
}

DEVICE = get_device("auto")
CONFIG_PATH = PROJECT_ROOT / "configs" / "resnet18.yaml"
CKPT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "resnet18_color_aug_best.pth"

# Load model
model = WoodNet(backbone="resnet18", num_classes=12, pretrained=False).to(DEVICE)
if CKPT_PATH.exists():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
model.eval()

eval_transform = build_eval_transform(224)
target_layer = get_target_layer(model)
gradcam_engine = GradCAM(model, target_layer)

def classify_wood(input_image, cam_alpha):
    if input_image is None:
        return {}, None, "Vui lòng chọn hoặc tải lên một ảnh mẫu."
    
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    input_image = input_image.convert("RGB")
    
    # Preprocess
    tensor = eval_transform(input_image).unsqueeze(0).to(DEVICE)
    
    # Grad-CAM heatmap
    heatmap = gradcam_engine(tensor, target_class=None)
    overlay = overlay_cam(input_image, heatmap, alpha=float(cam_alpha))
    
    # Forward logits
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    top_indices = np.argsort(probs)[::-1][:3]
    top_class = str(top_indices[0] + 1)
    
    confidences = {}
    for idx in top_indices:
        cls_key = str(idx + 1)
        name = f"{cls_key}. {WOOD_DETAILS[cls_key]['common_vi']} ({WOOD_DETAILS[cls_key]['scientific']})"
        confidences[name] = float(probs[idx])
        
    info = WOOD_DETAILS[top_class]
    md_output = f"""### 🏆 Loài Gỗ Dự Đoán: **{info['common_vi']}**
- **Tên khoa học:** *{info['scientific']}* ({info['common_en']})
- **Phân loại gỗ:** {info['type']}
- **Đặc trưng vân thớ:** {info['grain']}
- **Ứng dụng chế tác:** {info['usage']}
"""
    return confidences, overlay, md_output

# Collect sample images for gallery
examples = []
samples_dir = PROJECT_ROOT / "samples"
if samples_dir.exists():
    for i in range(1, 13):
        p = samples_dir / f"sample_{i}.jpg"
        if p.exists():
            examples.append([str(p), 0.5])

with gr.Blocks(title="Wood Grain Classifier - INT3401", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🪵 Hệ Thống Nhận Diện Loài Gỗ Cận Cảnh (Wood Grain Classification)")
    gr.Markdown("### BTL Học Máy (INT3401) - Lớp 11 | Mô hình Deep Transfer Learning (ResNet-18) + Grad-CAM")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Ảnh Macro Vân Gỗ Đầu Vào")
            alpha_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Độ trong suốt Grad-CAM Heatmap")
            btn = gr.Button("🔍 Phân Loại Loài Gỗ", variant="primary")
            
        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=3, label="Xác Suất Dự Đoán (Top-3)")
            cam_output = gr.Image(type="pil", label="Bản Đồ Nhiệt Grad-CAM")
            md_output = gr.Markdown(label="Thông Tin Loài Gỗ")
            
    btn.click(
        fn=classify_wood,
        inputs=[img_input, alpha_slider],
        outputs=[label_output, cam_output, md_output]
    )
    
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[img_input, alpha_slider],
            outputs=[label_output, cam_output, md_output],
            fn=classify_wood,
            cache_examples=False,
            label="📁 Danh Sách 12 Ảnh Mẫu Đại Diện Cho 12 Loài Gỗ"
        )

if __name__ == "__main__":
    demo.launch()
