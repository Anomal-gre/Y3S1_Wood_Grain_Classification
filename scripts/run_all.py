"""
Master reproduction script for INT3401 Wood Grain Classification.
Runs E1 (Baseline), E2 (ResNet18 Main), E6 (Ablation), E7 (Robustness), and E8 (Grad-CAM).
Saves comprehensive summary table and figures to outputs/.
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import set_seed, save_json, load_json
from scripts.download_dataset import main as run_download
from scripts.prepare_dataset import prepare_splits
from src.baseline_lbp_svm import run_baseline
from src.train import train
from src.evaluate import run_evaluation
from src.gradcam import generate_gradcam_gallery

def run_all(quick_mode: bool = False, skip_baseline: bool = False):
    set_seed(42)
    t_start = time.time()
    print("=" * 70)
    print("MASTER EXPERIMENT RUNNER: WOOD SPECIES TEXTURE CLASSIFICATION")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Quick mode: {quick_mode}")
    print("=" * 70)

    # 1. Dataset verification / download
    data_dir = PROJECT_ROOT / "dataset"
    splits_file = data_dir / "splits.json"
    if not splits_file.exists():
        print("\n>>> STEP 1: Downloading and preparing dataset...")
        run_download()
        prepare_splits(data_dir)
    else:
        print(f"\n>>> STEP 1: Using existing splits file at {splits_file}")

    # 2. Experiment E1: Baseline LBP + GLCM + SVM
    baseline_metrics = None
    if not skip_baseline:
        print("\n>>> STEP 2: Running Experiment E1 (Classical Texture Baseline)...")
        baseline_out = PROJECT_ROOT / "outputs" / "metrics" / "baseline_lbp_svm.json"
        baseline_metrics = run_baseline(
            splits_file=str(splits_file),
            output_file=str(baseline_out)
        )
    else:
        print("\n>>> STEP 2: Skipping baseline per user request.")

    # 3. Experiment E2: Main ResNet18 (Two-Stage Transfer Learning with Color Aug)
    print("\n>>> STEP 3: Running Experiment E2 (Main ResNet18 Two-Stage Training)...")
    cfg_main = PROJECT_ROOT / "configs" / "resnet18.yaml"
    train_res_main = train(str(cfg_main))

    print("\n>>> STEP 4: Evaluating ResNet18 and Running E7 (Robustness Test)...")
    eval_main = run_evaluation(str(cfg_main), run_robustness=True)

    # 4. Experiment E6: Ablation - ResNet18 WITHOUT Color Augmentation
    print("\n>>> STEP 5: Running Experiment E6 (Ablation: ResNet18 without Color Aug)...")
    cfg_no_color = PROJECT_ROOT / "configs" / "resnet18_no_color.yaml"
    train_res_no_color = train(str(cfg_no_color))
    eval_no_color = run_evaluation(str(cfg_no_color), run_robustness=True)

    # 5. Experiment E8: Grad-CAM Interpretability Gallery
    print("\n>>> STEP 6: Running Experiment E8 (Generating Grad-CAM Gallery)...")
    gradcam_fig = PROJECT_ROOT / "outputs" / "figures" / "gradcam_gallery.png"
    generate_gradcam_gallery(
        config_path=str(cfg_main),
        output_fig=str(gradcam_fig),
        num_correct=4,
        num_incorrect=4
    )

    # 6. Build Master Summary Table
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS INTO MASTER EXPERIMENT SUMMARY TABLE")
    print("=" * 70)

    summary_rows = []
    if baseline_metrics:
        summary_rows.append({
            "Mô hình": "Baseline LBP + GLCM + SVM",
            "Cấu hình": "RBF Kernel (Classical)",
            "Accuracy (%)": f"{baseline_metrics['test_accuracy']*100:.2f}%",
            "Macro-F1 (%)": f"{baseline_metrics['test_macro_f1']*100:.2f}%",
            "Độ trễ CPU (ms/ảnh)": f"{baseline_metrics['inference_latency_ms']:.2f} ms"
        })

    summary_rows.append({
        "Mô hình": "ResNet18 (Đề xuất)",
        "Cấu hình": "2-stage, Pretrained, Color Aug",
        "Accuracy (%)": f"{eval_main['accuracy']*100:.2f}%",
        "Macro-F1 (%)": f"{eval_main['macro_f1']*100:.2f}%",
        "Độ trễ CPU (ms/ảnh)": f"{eval_main['latency_ms']:.2f} ms"
    })

    summary_rows.append({
        "Mô hình": "ResNet18 (Ablation E6)",
        "Cấu hình": "2-stage, Pretrained, NO Color Aug",
        "Accuracy (%)": f"{eval_no_color['accuracy']*100:.2f}%",
        "Macro-F1 (%)": f"{eval_no_color['macro_f1']*100:.2f}%",
        "Độ trễ CPU (ms/ảnh)": f"{eval_no_color['latency_ms']:.2f} ms"
    })

    master_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runtime_minutes": round((time.time() - t_start) / 60, 2),
        "experiments": summary_rows,
        "robustness_comparison": {
            "resnet18_color_aug": eval_main.get("robustness_test", {}),
            "resnet18_no_color_aug": eval_no_color.get("robustness_test", {})
        }
    }

    out_summary_file = PROJECT_ROOT / "outputs" / "metrics" / "all_experiments_summary.json"
    save_json(master_summary, str(out_summary_file))

    # Write Markdown table
    md_table = "| Mô hình | Cấu hình | Accuracy | Macro-F1 | Độ trễ CPU |\n"
    md_table += "|---|---|---|---|---|\n"
    for r in summary_rows:
        md_table += f"| {r['Mô hình']} | {r['Cấu hình']} | {r['Accuracy (%)']} | {r['Macro-F1 (%)']} | {r['Độ trễ CPU (ms/ảnh)']} |\n"

    md_file = PROJECT_ROOT / "outputs" / "metrics" / "summary_table.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# TỔNG HỢP KẾT QUẢ THỰC NGHIỆM INT3401\n\n" + md_table)

    print("\n" + md_table)
    print(f"\nAll experiments finished successfully in {(time.time() - t_start)/60:.2f} minutes.")
    print(f"Summary JSON saved to: {out_summary_file}")
    print(f"Summary Markdown saved to: {md_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline LBP-SVM")
    args = parser.parse_args()
    run_all(skip_baseline=args.skip_baseline)
