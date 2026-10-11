import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from src.model import WoodNet
from src.dataset import build_eval_transform
from src.gradcam import GradCAM, overlay_cam, get_target_layer
STATIC_DIR = PROJECT_ROOT / "hf_space_static"

WOOD_DETAILS = {
    "1": {
        "id": 1,
        "scientific": "Fagus sylvatica",
        "common_en": "European Beech",
        "common_vi": "Gỗ Dẻ Gai Châu Âu",
        "type": "Gỗ cứng (Hardwood)",
        "porosity": "Lỗ mạch phân tán (Diffuse-porous)",
        "grain": "Thớ mịn, thẳng, tia gỗ ngắn dạng đốm nâu li ti phân bố đều khắp mặt cắt.",
        "usage": "Nội thất cao cấp, sàn gỗ chịu ma sát, đồ chơi mộc, cán công cụ kỹ thuật.",
        "density": "710 kg/m³"
    },
    "2": {
        "id": 2,
        "scientific": "Robinia pseudoacacia",
        "common_en": "Black Locust",
        "common_vi": "Gỗ Keo Dậu Hoa Trắng",
        "type": "Gỗ cứng (Hardwood cực bền)",
        "porosity": "Lỗ mạch dạng vòng (Ring-porous)",
        "grain": "Vòng năm rõ rệt, mạch gỗ xếp dạng vòng đồng tâm, vân nổi gợn sóng sắc nét.",
        "usage": "Trụ hàng rào ngoài trời, tà vẹt đường sắt, đóng tàu thuyền, sàn chịu tải nặng.",
        "density": "770 kg/m³"
    },
    "3": {
        "id": 3,
        "scientific": "Prunus avium",
        "common_en": "Wild Cherry",
        "common_vi": "Gỗ Anh Đào Dại (Cherry)",
        "type": "Gỗ cứng (Hardwood quý)",
        "porosity": "Lỗ mạch bán vòng (Semi-ring-porous)",
        "grain": "Vân mịn màng, sóng nhẹ lượn tự nhiên, chuyển ánh ấm đỏ sẫm sang trọng.",
        "usage": "Tủ kệ hoàng gia, đàn piano, violin, tẩu thuốc và đồ tiện mỹ nghệ cao cấp.",
        "density": "600 kg/m³"
    },
    "4": {
        "id": 4,
        "scientific": "Castanea sativa",
        "common_en": "Sweet Chestnut",
        "common_vi": "Gỗ Dẻ Trùng Khánh (Castanea)",
        "type": "Gỗ cứng (Hardwood)",
        "porosity": "Lỗ mạch dạng vòng (Ring-porous)",
        "grain": "Mạch gỗ thô dạng vòng tương tự sồi nhưng không có tia gỗ nổi hạt vảy bạc.",
        "usage": "Thùng ủ rượu vang truyền thống, cột kèo nhà cổ, đồ gỗ ngoài trời chống mục.",
        "density": "560 kg/m³"
    },
    "5": {
        "id": 5,
        "scientific": "Pseudotsuga menziesii",
        "common_en": "Douglas Fir",
        "common_vi": "Gỗ Thông Douglas",
        "type": "Gỗ mềm (Softwood lá kim)",
        "porosity": "Gỗ không lỗ mạch (Non-porous/Gymnosperm)",
        "grain": "Vòng sinh trưởng mùa xuân/thu tương phản cực mạnh, vân dải sọc thẳng tắp.",
        "usage": "Kết cấu xây dựng chịu lực, xà gồ nhịp lớn, ván ép cốp pha chất lượng cao.",
        "density": "530 kg/m³"
    },
    "6": {
        "id": 6,
        "scientific": "Larix decidua",
        "common_en": "European Larch",
        "common_vi": "Gỗ Thông Rụng Lá Châu Âu",
        "type": "Gỗ mềm (Softwood chịu nước)",
        "porosity": "Gỗ lá kim (Gymnosperm)",
        "grain": "Vân dạng dải hẹp, màu nâu đỏ đậm dần về tâm, rất giàu tinh dầu nhựa tự nhiên.",
        "usage": "Ốp tường ngoại thất chịu mưa nắng, du thuyền, cầu cảng, ván sàn ban công.",
        "density": "590 kg/m³"
    },
    "7": {
        "id": 7,
        "scientific": "Quercus robur",
        "common_en": "English Oak",
        "common_vi": "Gỗ Sồi Châu Âu (Sồi Trắng)",
        "type": "Gỗ cứng (Hardwood danh tiếng)",
        "porosity": "Lỗ mạch dạng vòng (Ring-porous)",
        "grain": "Tia gỗ lớn đặc trưng dạng vệt bạc (medullary rays), tom gỗ thô sang trọng.",
        "usage": "Nội thất hoàng gia, sàn sồi tự nhiên, thùng ủ whisky và cognac trứ danh.",
        "density": "720 kg/m³"
    },
    "8": {
        "id": 8,
        "scientific": "Pinus sylvestris",
        "common_en": "Scots Pine",
        "common_vi": "Gỗ Thông Scots",
        "type": "Gỗ mềm (Softwood phổ biến)",
        "porosity": "Gỗ lá kim có ống nhựa (Resin canals)",
        "grain": "Vân xoắn nhẹ, có mắt gỗ tròn tự nhiên, viền mùa muộn màu nâu sẫm rõ ràng.",
        "usage": "Đồ nội thất gia đình Scandinavian, thùng bao bì mộc, cấu kiện nhà gỗ.",
        "density": "510 kg/m³"
    },
    "9": {
        "id": 9,
        "scientific": "Platanus orientalis",
        "common_en": "Oriental Plane",
        "common_vi": "Gỗ Ti Trăn Phương Đông",
        "type": "Gỗ cứng (Hardwood)",
        "porosity": "Lỗ mạch phân tán (Diffuse-porous)",
        "grain": "Vân đốm vảy báo (lacewood / ray flecks) lấp lánh phản quang khi xẻ tiếp tuyến.",
        "usage": "Veneer trang trí siêu xe, báng súng thể thao, cần đàn guitar, đồ khảm xà cừ.",
        "density": "620 kg/m³"
    },
    "10": {
        "id": 10,
        "scientific": "Populus tremula",
        "common_en": "Aspen / Poplar",
        "common_vi": "Gỗ Bạch Dương Rung (Aspen)",
        "type": "Gỗ cứng mềm (Soft Hardwood)",
        "porosity": "Lỗ mạch phân tán rất mịn",
        "grain": "Vân mờ nhạt, đồng nhất, bề mặt thớ mịn xốp, không có mùi nhựa hay tannin.",
        "usage": "Phòng xông hơi Sauna (cách nhiệt, không bỏng da), que diêm, đũa ăn, bột giấy.",
        "density": "450 kg/m³"
    },
    "11": {
        "id": 11,
        "scientific": "Ailanthus altissima",
        "common_en": "Tree of Heaven",
        "common_vi": "Gỗ Cây Thiên Đường (Ailanthus)",
        "type": "Gỗ cứng (Hardwood)",
        "porosity": "Lỗ mạch dạng vòng thô",
        "grain": "Vân thô ráp, lỗ mạch lớn tương tự tần bì (ash), bề mặt gân guốc khỏe khoắn.",
        "usage": "Đồ mộc gia dụng, đồ gỗ trang trí rustic, pallet chịu tải, tay vịn cầu thang.",
        "density": "580 kg/m³"
    },
    "12": {
        "id": 12,
        "scientific": "Juglans regia",
        "common_en": "Persian Walnut",
        "common_vi": "Gỗ Óc Chó Phương Tây (Walnut)",
        "type": "Gỗ cứng (Hardwood quý tộc)",
        "porosity": "Lỗ mạch bán vòng (Semi-ring-porous)",
        "grain": "Vân cuộn sóng, xoáy ảo diệu, dải màu socola từ nâu socola nhạt đến tím sẫm.",
        "usage": "Nội thất siêu xe Rolls-Royce/Bentley, báng súng săn cao cấp, bàn hội nghị.",
        "density": "640 kg/m³"
    }
}

def main():
    print("[*] Setting up directories...")
    samples_out = STATIC_DIR / "assets" / "samples"
    cams_out = STATIC_DIR / "assets" / "cams"
    figures_out = STATIC_DIR / "assets" / "figures"
    
    samples_out.mkdir(parents=True, exist_ok=True)
    cams_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)
    
    # Copy evaluation figures
    fig_src = PROJECT_ROOT / "outputs" / "figures"
    for fig_name in [
        "resnet18_color_aug_confusion_matrix.png",
        "resnet18_color_aug_learning_curves.png",
        "resnet18_color_aug_robustness_test.png",
        "gradcam_gallery.png"
    ]:
        if (fig_src / fig_name).exists():
            shutil.copy2(fig_src / fig_name, figures_out / fig_name)
            print(f"  [+] Copied figure {fig_name}")
            
    # Load model for Grad-CAM generation
    print("[*] Loading ResNet18 model...")
    model = WoodNet(backbone="resnet18", num_classes=12, pretrained=False)
    ckpt_path = PROJECT_ROOT / "outputs" / "checkpoints" / "resnet18_color_aug_best.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    eval_tf = build_eval_transform(224)
    target_layer = get_target_layer(model)
    cam_engine = GradCAM(model, target_layer)
    
    samples_data = []
    
    for i in range(1, 13):
        sample_path = PROJECT_ROOT / "samples" / f"sample_{i}.jpg"
        img = Image.open(sample_path).convert("RGB")
        # Save sample
        img.save(samples_out / f"sample_{i}.jpg", quality=92)
        
        # Grad-CAM heatmap
        tensor = eval_tf(img).unsqueeze(0)
        heatmap = cam_engine.generate(tensor, target_class=i-1)
        cam_img = overlay_cam(img, heatmap, alpha=0.55)
        if isinstance(cam_img, np.ndarray):
            cam_pil = Image.fromarray(cam_img)
        else:
            cam_pil = cam_img
        cam_pil.save(cams_out / f"cam_{i}.jpg", quality=92)
        
        # Inference
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).numpy()
            
        top_indices = np.argsort(probs)[::-1][:4]
        
        predictions = []
        for idx in top_indices:
            cls_k = str(idx + 1)
            predictions.append({
                "class_id": int(cls_k),
                "name": WOOD_DETAILS[cls_k]["common_vi"],
                "scientific": WOOD_DETAILS[cls_k]["scientific"],
                "probability": float(probs[idx])
            })
            
        species_info = WOOD_DETAILS[str(i)]
        samples_data.append({
            "id": i,
            "filename": f"sample_{i}.jpg",
            "cam_filename": f"cam_{i}.jpg",
            "species": species_info,
            "predictions": predictions,
            "is_correct": (predictions[0]["class_id"] == i),
            "confidence": predictions[0]["probability"]
        })
        print(f"  [+] Processed species {i}: {species_info['common_vi']} (Conf: {predictions[0]['probability']*100:.1f}%)")

    # Metrics summary
    app_data = {
        "title": "Wood Grain Species Classification & Visual Interpretability",
        "course": "Học Máy (INT3401) - Lớp 11",
        "team": "Nhóm 11",
        "model": "ResNet-18 Deep Transfer Learning (Two-Stage Training + Color Augmentation)",
        "dataset": "WOOD-AUTH Benchmark (Barmpoutis 2018), 8,544 ảnh microscopic wood grain",
        "kpis": {
            "test_accuracy": "92.74%",
            "macro_f1": "92.57%",
            "weighted_f1": "92.75%",
            "num_classes": 12,
            "latency_cpu": "14.73 ms",
            "val_f1_best": "92.82%"
        },
        "baseline_comparison": [
            {"model": "Baseline LBP + GLCM + RBF-SVM", "f1": 78.92, "acc": 80.99, "latency": "0.36 ms"},
            {"model": "ResNet-18 (Stage 1 Warmup)", "f1": 85.40, "acc": 86.10, "latency": "14.7 ms"},
            {"model": "ResNet-18 Two-Stage (Đề xuất)", "f1": 92.57, "acc": 92.74, "latency": "14.7 ms"}
        ],
        "robustness_benchmark": [
            {"condition": "Ảnh nguyên bản (Original)", "f1": 92.57},
            {"condition": "Mất màu hoàn toàn (Grayscale)", "f1": 82.06},
            {"condition": "Giảm bão hòa màu 50% (Desaturate)", "f1": 88.82},
            {"condition": "Lệch tông màu +20% (Hue +0.2)", "f1": 78.90},
            {"condition": "Lệch tông màu -20% (Hue -0.2)", "f1": 82.96},
            {"condition": "Thiếu sáng (Dark / Low-light)", "f1": 90.74},
            {"condition": "Thừa sáng (Bright / Over-exposed)", "f1": 91.04}
        ],
        "samples": samples_data
    }
    
    with open(STATIC_DIR / "data.json", "w", encoding="utf-8") as f:
        json.dump(app_data, f, ensure_ascii=False, indent=2)
    print(f"[*] Saved data.json with {len(samples_data)} wood species data.")

if __name__ == "__main__":
    main()
