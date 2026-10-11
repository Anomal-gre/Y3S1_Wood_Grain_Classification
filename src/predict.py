"""
Single-image inference CLI for Wood Species Classification with Grad-CAM visualization.
"""

import sys
import argparse
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import torch
import numpy as np

from src.utils import load_config, get_device
from src.dataset import build_eval_transform, load_split_samples
from src.model import WoodNet
from src.gradcam import GradCAM, overlay_cam, get_target_layer

def predict_single_image(
    image_path: str,
    config_path: str = "configs/resnet18.yaml",
    ckpt_path: str = None,
    output_vis: str = None,
    top_k: int = 3,
    device_name: str = "auto"
):
    cfg = load_config(config_path)
    exp_name = cfg.get("experiment_name", "wood_exp")
    device = get_device(device_name)

    if ckpt_path is None:
        ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / f"{exp_name}_best.pth"
    else:
        ckpt_path = Path(ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=device)
    metadata = checkpoint.get("metadata", {})
    class_dict = metadata.get("classes", {})

    model = WoodNet(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    eval_tf = build_eval_transform(img_size=cfg["data"].get("img_size", 224))
    with Image.open(image_path) as im:
        im_rgb = im.convert("RGB")
        tensor = eval_tf(im_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probs)[::-1][:top_k]

    print("\n" + "=" * 50)
    print(f"PREDICTION RESULTS FOR: {Path(image_path).name}")
    print("=" * 50)
    for rank, idx in enumerate(top_indices, 1):
        cls_key = str(idx + 1)
        cls_info = class_dict.get(cls_key, {"name": f"Species {cls_key}", "vietnamese": ""})
        prob_pct = probs[idx] * 100
        print(f"Top {rank}: {cls_info['name']} ({cls_info.get('vietnamese', '')}) - {prob_pct:.2f}%")

    if output_vis:
        target_layer = get_target_layer(model)
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate(tensor, target_class=int(top_indices[0]))
        overlay = overlay_cam(im_rgb, cam)
        Image.fromarray(overlay).save(output_vis)
        print(f"\nSaved Grad-CAM overlay to: {output_vis}")

    return [(class_dict.get(str(i+1), {}), float(probs[i])) for i in top_indices]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict wood species from grain image")
    parser.add_argument("--image", type=str, required=True, help="Path to macroscopic wood image")
    parser.add_argument("--config", type=str, default="configs/resnet18.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--vis", type=str, default=None, help="Path to save Grad-CAM overlay")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    predict_single_image(
        image_path=args.image,
        config_path=args.config,
        ckpt_path=args.ckpt,
        output_vis=args.vis,
        top_k=args.top_k
    )
