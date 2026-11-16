"""Shared, deterministic inference for the CLI and Streamlit demo."""
from pathlib import Path

import torch

from src.dataset import build_eval_transform, build_eval_image_transform
from src.model import WoodNet
from src.species import SPECIES_MAPPING
from src.utils import get_device, load_config

def load_predictor(config_path="configs/resnet18.yaml", ckpt_path=None, device_name="cpu"):
    cfg = load_config(config_path)
    if ckpt_path is None:
        ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / f"{cfg['experiment_name']}_best.pth"
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Chưa có checkpoint: {ckpt_path}. Huấn luyện bằng "
            f"python src/train.py --config {config_path} trước khi dự đoán."
        )
    device = get_device(device_name)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = WoodNet(backbone=cfg["model"]["backbone"],
                    num_classes=cfg["model"]["num_classes"], pretrained=False,
                    dropout_rate=cfg["model"].get("dropout_rate", 0.4)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg, device


def predict_image(model, cfg, device, image, top_k=3):
    if not 1 <= top_k <= cfg["model"]["num_classes"]:
        raise ValueError("top_k phải nằm trong số lớp của mô hình.")
    size = cfg["data"]["img_size"]
    model.eval()
    image = image.convert("RGB")
    tensor = build_eval_transform(size)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = torch.softmax(model(tensor), dim=1)[0]
    values, indices = scores.topk(top_k)
    results = [(SPECIES_MAPPING[str(index + 1)], float(value))
               for index, value in zip(indices.tolist(), values.tolist())]
    # Use the same geometric view, before normalization, for display and Grad-CAM.
    model_view = build_eval_image_transform(size)(image)
    return results, tensor, model_view
