"""Single-image inference with optional, spatially aligned Grad-CAM."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from PIL import Image
from src.gradcam import GradCAM, get_target_layer, overlay_cam
from src.inference import load_predictor, predict_image

def predict_single_image(image_path, config_path="configs/resnet18.yaml",
                         ckpt_path=None, output_vis=None, top_k=3, device_name="cpu"):
    model, cfg, device = load_predictor(config_path, ckpt_path, device_name)
    with Image.open(image_path) as source:
        results, tensor, model_view = predict_image(model, cfg, device, source, top_k)
    for rank, (species, score) in enumerate(results, 1):
        print(f"Top {rank}: {species['vietnamese']} / {species['name']} — {score * 100:.2f}%")
    if output_vis:
        cam = GradCAM(model, get_target_layer(model))
        try:
            heatmap = cam.generate(tensor, results[0][0]["id"] - 1)
            destination = Path(output_vis)
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay_cam(model_view, heatmap)).save(destination)
            print(f"Saved Grad-CAM: {destination}")
        finally:
            cam.close()
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--vis", default=None)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    predict_single_image(args.image, args.config, args.ckpt, args.vis, args.top_k, args.device)
