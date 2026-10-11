"""
Evaluation, Confusion Matrix, Latency Benchmarking, and Color Robustness Testing (E7).
"""

import time
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

from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.utils import set_seed, load_config, save_json, get_device
from src.dataset import (
    load_split_samples,
    build_eval_transform,
    get_dataloader,
    COLOR_SHIFTS
)
from src.model import WoodNet

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = True
):
    """Plot and save high-resolution confusion matrix heatmap."""
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True
    )
    plt.title("Normalized Confusion Matrix on Test Set", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Run evaluation and collect predictions, targets, probabilities, and latency."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    total_time = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            t0 = time.time()
            outputs = model(images)
            total_time += time.time() - t0

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

    num_samples = len(all_targets)
    ms_per_image = (total_time / max(1, num_samples)) * 1000.0

    return np.array(all_preds), np.array(all_targets), np.array(all_probs), ms_per_image

def run_evaluation(
    config_path: str,
    ckpt_path: str = None,
    run_robustness: bool = True,
    device_name: str = "auto"
):
    cfg = load_config(config_path)
    exp_name = cfg.get("experiment_name", "wood_exp")
    device = get_device(device_name)
    print("=" * 60)
    print(f"Evaluating Model: {exp_name} on Test Set")
    print(f"Device: {device}")
    print("=" * 60)

    # 1. Determine checkpoint path
    if ckpt_path is None:
        ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / f"{exp_name}_best.pth"
    else:
        ckpt_path = Path(ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 2. Load model
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_cfg = cfg["model"]
    model = WoodNet(
        backbone=model_cfg.get("backbone", "resnet18"),
        num_classes=model_cfg.get("num_classes", 12),
        pretrained=False,
        dropout_rate=model_cfg.get("dropout_rate", 0.4)
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Load Test Data
    splits_file = cfg["data"]["splits_file"]
    test_samples, metadata = load_split_samples(splits_file, "test")
    print(f"Loaded {len(test_samples)} test samples from {splits_file}")

    class_dict = metadata.get("classes", {})
    class_names = [class_dict[str(i+1)]["name"] for i in range(len(class_dict))] if class_dict else [f"Class {i+1}" for i in range(12)]

    img_size = cfg["data"].get("img_size", 224)
    eval_tf = build_eval_transform(img_size=img_size)

    test_loader = get_dataloader(
        test_samples,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        transform=eval_tf
    )

    # 4. Standard Test Evaluation
    preds, targets, probs, latency_ms = evaluate_model(model, test_loader, device)

    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets, preds, average="weighted", zero_division=0))
    report_dict = classification_report(targets, preds, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(targets, preds, target_names=class_names, zero_division=0)

    print("\n" + "=" * 50)
    print("TEST BENCHMARK RESULTS:")
    print(f"Accuracy:         {acc * 100:.2f}%")
    print(f"Macro-F1:         {macro_f1 * 100:.2f}%")
    print(f"Weighted-F1:      {weighted_f1 * 100:.2f}%")
    print(f"Inference latency:{latency_ms:.2f} ms/image ({device})")
    print("=" * 50)
    print("\nPer-class Classification Report:\n")
    print(report_text)

    # 5. Save Confusion Matrix
    cm = confusion_matrix(targets, preds)
    cm_path = Path(cfg["output"]["figures_dir"]) / f"{exp_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, str(cm_path), normalize=True)
    print(f"Saved confusion matrix figure to: {cm_path}")

    metrics_dir = Path(cfg["output"]["metrics_dir"])
    eval_metrics = {
        "experiment": exp_name,
        "checkpoint": str(ckpt_path),
        "test_samples": len(test_samples),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "latency_ms": round(latency_ms, 2),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }

    # 6. Experiment E7: Color Robustness Testing
    if run_robustness:
        print("\n" + "=" * 50)
        print("RUNNING EXPERIMENT E7: Color Robustness Test")
        print("Testing robustness across 7 color perturbations...")
        print("=" * 50)

        robustness_results = {}
        for shift_name in COLOR_SHIFTS.keys():
            shift_loader = get_dataloader(
                test_samples,
                batch_size=64,
                shuffle=False,
                num_workers=0,
                transform=eval_tf,
                color_shift_name=shift_name
            )
            s_preds, s_targets, _, _ = evaluate_model(model, shift_loader, device)
            s_acc = float(accuracy_score(s_targets, s_preds))
            s_f1 = float(f1_score(s_targets, s_preds, average="macro", zero_division=0))
            robustness_results[shift_name] = {
                "accuracy": round(s_acc, 4),
                "macro_f1": round(s_f1, 4),
                "acc_drop": round((acc - s_acc) * 100, 2),
                "f1_drop": round((macro_f1 - s_f1) * 100, 2)
            }
            print(f"[{shift_name:12s}] Acc: {s_acc*100:6.2f}% | Macro-F1: {s_f1*100:6.2f}% | F1 Drop: {(macro_f1 - s_f1)*100:+5.2f}%")

        eval_metrics["robustness_test"] = robustness_results

        # Plot Robustness Bar Chart
        fig, ax = plt.subplots(figsize=(10, 5))
        shifts = list(robustness_results.keys())
        f1_vals = [robustness_results[s]["macro_f1"] * 100 for s in shifts]
        colors = ["#1f77b4" if s == "original" else "#ff7f0e" for s in shifts]
        bars = ax.bar(shifts, f1_vals, color=colors, width=0.6, edgecolor="black", alpha=0.85)
        ax.set_ylabel("Macro-F1 (%)", fontsize=11)
        ax.set_title(f"E7 Robustness Test: Performance under Color Perturbations ({exp_name})", fontsize=12)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        rob_fig_path = Path(cfg["output"]["figures_dir"]) / f"{exp_name}_robustness_test.png"
        plt.savefig(rob_fig_path, dpi=300)
        plt.close()
        print(f"Saved robustness plot to: {rob_fig_path}")

    # Save metrics JSON
    metrics_file = metrics_dir / f"{exp_name}_test_metrics.json"
    save_json(eval_metrics, str(metrics_file))
    print(f"Saved test metrics JSON to: {metrics_file}")

    return eval_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Wood Grain Classifier")
    parser.add_argument("--config", type=str, default="configs/resnet18.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--no-robustness", action="store_true", help="Skip robustness test")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        ckpt_path=args.ckpt,
        run_robustness=not args.no_robustness,
        device_name=args.device
    )
