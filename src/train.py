"""
Two-Stage Transfer Learning Trainer for Wood Species Classification.
Implements Stage 1 Head Warm-up, Stage 2 Full Fine-tuning with Differential LRs,
Cosine Annealing scheduler, Early Stopping, and detailed metrics logging.
"""

import os
import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from src.utils import set_seed, load_config, setup_logger, save_json, get_device
from src.dataset import (
    load_split_samples,
    build_train_transform,
    build_eval_transform,
    get_dataloader
)
from src.model import WoodNet

def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_train: bool,
    backbone_frozen: bool
) -> Tuple[float, float, float]:
    """Execute one training or evaluation epoch."""
    if is_train:
        model.train()
        if backbone_frozen:
            # Maintain BatchNorm running statistics in frozen backbone
            model.backbone.eval()
    else:
        model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(is_train):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            if backbone_frozen:
                with torch.no_grad():
                    feat = model.backbone(images)
                outputs = model.head(feat)
            else:
                outputs = model(images)

            loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())

    num_samples = len(all_targets)
    epoch_loss = total_loss / max(1, num_samples)
    epoch_acc = float(accuracy_score(all_targets, all_preds))
    epoch_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

    return epoch_loss, epoch_acc, epoch_f1

def plot_learning_curves(history: Dict[str, List], save_path: str):
    """Plot and save loss and Macro-F1 curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Macro-F1 plot
    ax1_f1 = history["train_f1"]
    ax2.plot(epochs, ax1_f1, label="Train Macro-F1", color="#2ca02c", lw=2)
    ax2.plot(epochs, history["val_f1"], label="Val Macro-F1", color="#d62728", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.set_title("Training & Validation Macro-F1")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def train(config_path: str, override_seed: int = None, override_device: str = None):
    cfg = load_config(config_path)
    seed = override_seed if override_seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    torch.set_num_threads(os.cpu_count() or 6)

    exp_name = cfg.get("experiment_name", "wood_exp")
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    metrics_dir = Path(cfg["output"]["metrics_dir"])
    figures_dir = Path(cfg["output"]["figures_dir"])
    for d in [ckpt_dir, metrics_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_file = metrics_dir / f"{exp_name}_train.log"
    logger = setup_logger(exp_name, str(log_file))
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Random seed set to {seed}")

    device = get_device(override_device or cfg["train"].get("device", "auto"))
    logger.info(f"Using compute device: {device}")

    # 1. Load data
    splits_file = cfg["data"]["splits_file"]
    train_samples, metadata = load_split_samples(splits_file, "train")
    val_samples, _ = load_split_samples(splits_file, "val")
    logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    img_size = cfg["data"].get("img_size", 224)
    color_aug = cfg["data"].get("color_aug", True)
    batch_size = cfg["data"].get("batch_size", 64)
    num_workers = cfg["data"].get("num_workers", 0)

    train_tf = build_train_transform(img_size=img_size, color_aug=color_aug)
    eval_tf = build_eval_transform(img_size=img_size)

    train_loader = get_dataloader(
        train_samples,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        transform=train_tf,
        drop_last=True
    )
    val_loader = get_dataloader(
        val_samples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        transform=eval_tf,
        drop_last=False
    )

    # 2. Build Model
    model_cfg = cfg["model"]
    num_classes = model_cfg.get("num_classes", 12)
    backbone_name = model_cfg.get("backbone", "resnet18")
    pretrained = model_cfg.get("pretrained", True)
    dropout_rate = model_cfg.get("dropout_rate", 0.4)

    model = WoodNet(
        backbone=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    ).to(device)

    # Loss function with label smoothing
    label_smoothing = cfg["train"].get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    history = {
        "epoch": [],
        "stage": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "lr": []
    }

    best_val_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = ckpt_dir / f"{exp_name}_best.pth"
    patience = cfg["train"].get("early_stopping_patience", 5)
    epochs_no_improve = 0

    total_start_time = time.time()

    # ==========================================
    # STAGE 1: Head Warm-up (Backbone Frozen)
    # ==========================================
    epochs_stage1 = cfg["train"].get("epochs_stage1", 5)
    lr_stage1 = cfg["train"].get("lr_stage1", 1e-3)
    weight_decay = cfg["train"].get("weight_decay", 1e-2)

    logger.info("=" * 60)
    logger.info(f"STARTING STAGE 1: Head Warm-up ({epochs_stage1} epochs, backbone frozen)")
    logger.info("=" * 60)

    model.freeze_backbone()
    optimizer_s1 = torch.optim.AdamW(model.head.parameters(), lr=lr_stage1, weight_decay=weight_decay)

    for ep in range(1, epochs_stage1 + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer_s1, device, is_train=True, backbone_frozen=True
        )
        va_loss, va_acc, va_f1 = run_epoch(
            model, val_loader, criterion, optimizer_s1, device, is_train=False, backbone_frozen=True
        )
        t_ep = time.time() - t0

        history["epoch"].append(ep)
        history["stage"].append(1)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["train_f1"].append(round(tr_f1, 4))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc, 4))
        history["val_f1"].append(round(va_f1, 4))
        history["lr"].append(lr_stage1)

        logger.info(
            f"Stage 1 [Epoch {ep}/{epochs_stage1}] ({t_ep:.1f}s) - "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.2f}%, F1: {tr_f1*100:.2f}% | "
            f"Val Loss: {va_loss:.4f}, Acc: {va_acc*100:.2f}%, F1: {va_f1*100:.2f}%"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = ep
            torch.save({
                "epoch": ep,
                "stage": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_s1.state_dict(),
                "val_f1": va_f1,
                "val_acc": va_acc,
                "config": cfg,
                "metadata": metadata
            }, best_ckpt_path)
            logger.info(f" -> Checkpoint saved (Val Macro-F1: {va_f1*100:.2f}%)")

    # ==========================================
    # STAGE 2: Full Fine-tuning (Backbone Unfrozen)
    # ==========================================
    epochs_stage2 = cfg["train"].get("epochs_stage2", 15)
    lr_backbone = cfg["train"].get("lr_backbone_stage2", 1e-4)
    lr_head = cfg["train"].get("lr_head_stage2", 1e-3)

    logger.info("=" * 60)
    logger.info(f"STARTING STAGE 2: Full Fine-Tuning ({epochs_stage2} epochs, backbone unfrozen)")
    logger.info(f"Differential LRs: Backbone = {lr_backbone}, Head = {lr_head}")
    logger.info("=" * 60)

    model.unfreeze_backbone()
    param_groups = model.get_parameter_groups(lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay)
    optimizer_s2 = torch.optim.AdamW(param_groups)
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=epochs_stage2)

    for ep in range(epochs_stage1 + 1, epochs_stage1 + epochs_stage2 + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer_s2, device, is_train=True, backbone_frozen=False
        )
        va_loss, va_acc, va_f1 = run_epoch(
            model, val_loader, criterion, optimizer_s2, device, is_train=False, backbone_frozen=False
        )
        scheduler_s2.step()
        t_ep = time.time() - t0

        curr_lr = optimizer_s2.param_groups[0]["lr"]
        history["epoch"].append(ep)
        history["stage"].append(2)
        history["train_loss"].append(round(tr_loss, 4))
        history["train_acc"].append(round(tr_acc, 4))
        history["train_f1"].append(round(tr_f1, 4))
        history["val_loss"].append(round(va_loss, 4))
        history["val_acc"].append(round(va_acc, 4))
        history["val_f1"].append(round(va_f1, 4))
        history["lr"].append(curr_lr)

        logger.info(
            f"Stage 2 [Epoch {ep}/{epochs_stage1 + epochs_stage2}] ({t_ep:.1f}s) - "
            f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc*100:.2f}%, F1: {tr_f1*100:.2f}% | "
            f"Val Loss: {va_loss:.4f}, Acc: {va_acc*100:.2f}%, F1: {va_f1*100:.2f}%"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = ep
            epochs_no_improve = 0
            torch.save({
                "epoch": ep,
                "stage": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_s2.state_dict(),
                "val_f1": va_f1,
                "val_acc": va_acc,
                "config": cfg,
                "metadata": metadata
            }, best_ckpt_path)
            logger.info(f" -> Checkpoint saved (New Best Val Macro-F1: {va_f1*100:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                break

    total_duration = time.time() - total_start_time
    logger.info("=" * 60)
    logger.info(f"Training completed in {total_duration/60:.2f} minutes.")
    logger.info(f"Best Validation Macro-F1: {best_val_f1*100:.2f}% achieved at epoch {best_epoch}.")
    logger.info(f"Best model saved to: {best_ckpt_path}")
    logger.info("=" * 60)

    # Save history and curves
    history_file = metrics_dir / f"{exp_name}_history.json"
    save_json(history, str(history_file))
    curve_file = figures_dir / f"{exp_name}_learning_curves.png"
    plot_learning_curves(history, str(curve_file))
    logger.info(f"Saved learning curves to {curve_file}")

    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_ckpt_path),
        "history_file": str(history_file),
        "curve_file": str(curve_file)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wood Grain Classifier")
    parser.add_argument("--config", type=str, default="configs/resnet18.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--device", type=str, default=None, help="Compute device override ('cpu' or 'cuda')")
    args = parser.parse_args()

    train(config_path=args.config, override_seed=args.seed, override_device=args.device)
