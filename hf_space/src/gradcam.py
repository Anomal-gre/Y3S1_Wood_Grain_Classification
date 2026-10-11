"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Model Interpretability (E8).
Generates saliency heatmaps for both correctly classified and misclassified wood grain samples.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import set_seed, load_config, get_device
from src.dataset import build_eval_transform, load_split_samples
from src.model import WoodNet

class GradCAM:
    """Grad-CAM implementation for convolutional vision backbones."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]
        self.model.zero_grad()
        score.backward()

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of forward activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        # Apply ReLU
        cam = F.relu(cam)

        # Normalize between 0 and 1
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam(img_pil: Image.Image, cam_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay CAM heatmap on top of PIL image."""
    w, h = img_pil.size
    cam_resized = cv2.resize(cam_map, (w, h))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_np = np.array(img_pil.convert("RGB"))
    blended = (alpha * heatmap_color + (1.0 - alpha) * img_np).astype(np.uint8)
    return blended

def get_target_layer(model: WoodNet) -> nn.Module:
    """Extract the final convolutional layer of the backbone."""
    if hasattr(model.backbone, "layer4"):
        return model.backbone.layer4[-1]
    elif hasattr(model.backbone, "features"):
        return model.backbone.features[-1]
    raise ValueError(f"Unknown backbone structure for {model.backbone_name}")

def generate_gradcam_gallery(
    config_path: str = "configs/resnet18.yaml",
    ckpt_path: str = None,
    output_fig: str = "outputs/figures/gradcam_gallery.png",
    num_correct: int = 4,
    num_incorrect: int = 4,
    device_name: str = "auto"
):
    """
    Find correctly classified and misclassified samples on Test set,
    and generate a comprehensive Grad-CAM visualization gallery.
    """
    cfg = load_config(config_path)
    exp_name = cfg.get("experiment_name", "wood_exp")
    device = get_device(device_name)

    if ckpt_path is None:
        ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / f"{exp_name}_best.pth"
    else:
        ckpt_path = Path(ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = WoodNet(
        backbone=cfg["model"]["backbone"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_layer = get_target_layer(model)
    gradcam = GradCAM(model, target_layer)

    test_samples, metadata = load_split_samples(cfg["data"]["splits_file"], "test")
    class_dict = metadata.get("classes", {})
    class_names = [class_dict[str(i+1)]["name"] for i in range(len(class_dict))]

    eval_tf = build_eval_transform(img_size=cfg["data"].get("img_size", 224))

    correct_samples = []
    incorrect_samples = []

    print("Searching for diverse correct and incorrect test samples...")
    for img_path, gt_label in test_samples:
        with Image.open(img_path) as im:
            im_rgb = im.convert("RGB")
            tensor = eval_tf(im_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_label = int(np.argmax(prob))
            conf = float(prob[pred_label])

        sample_info = {
            "path": img_path,
            "image": im_rgb,
            "tensor": tensor,
            "gt": gt_label,
            "pred": pred_label,
            "conf": conf
        }

        if pred_label == gt_label and len(correct_samples) < num_correct:
            correct_samples.append(sample_info)
        elif pred_label != gt_label and len(incorrect_samples) < num_incorrect:
            incorrect_samples.append(sample_info)

        if len(correct_samples) >= num_correct and len(incorrect_samples) >= num_incorrect:
            break

    all_show = [("Correct Prediction", s) for s in correct_samples] + [("Misclassified Sample", s) for s in incorrect_samples]
    total_samples = len(all_show)

    fig, axes = plt.subplots(total_samples, 3, figsize=(12, 3.2 * total_samples))
    if total_samples == 1:
        axes = np.expand_dims(axes, 0)

    for idx, (category_title, sample) in enumerate(all_show):
        cam = gradcam.generate(sample["tensor"], target_class=sample["pred"])
        overlay = overlay_cam(sample["image"], cam)

        gt_name = class_names[sample["gt"]]
        pred_name = class_names[sample["pred"]]
        conf_pct = sample["conf"] * 100

        # Col 1: Original Image
        axes[idx, 0].imshow(sample["image"])
        axes[idx, 0].set_title(f"[{category_title}]\nGround Truth: {gt_name}", fontsize=10)
        axes[idx, 0].axis("off")

        # Col 2: Heatmap
        axes[idx, 1].imshow(cam, cmap="jet")
        axes[idx, 1].set_title("Grad-CAM Activation Map", fontsize=10)
        axes[idx, 1].axis("off")

        # Col 3: Overlay
        axes[idx, 2].imshow(overlay)
        color = "green" if sample["gt"] == sample["pred"] else "red"
        axes[idx, 2].set_title(f"Predicted: {pred_name} ({conf_pct:.1f}%)", fontsize=10, color=color, fontweight="bold")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    Path(output_fig).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=300)
    plt.close()
    print(f"Successfully generated Grad-CAM gallery at: {output_fig}")

if __name__ == "__main__":
    generate_gradcam_gallery()
