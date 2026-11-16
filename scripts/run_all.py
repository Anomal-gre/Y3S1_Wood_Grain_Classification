"""Reproduce the core experiments; unrecorded ablation is explicitly opt-in."""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def run_all(skip_baseline=False, ablation=False):
    # Config paths in the project are relative to its root.
    os.chdir(ROOT)
    from scripts.download_dataset import main as download
    from scripts.prepare_dataset import prepare_splits
    from src.baseline_lbp_svm import run_baseline
    from src.train import train
    from src.evaluate import run_evaluation
    from src.gradcam import generate_gradcam_gallery
    from src.utils import save_json
    started = time.perf_counter()
    if not (ROOT / "dataset/splits.json").is_file():
        download()
        prepare_splits(ROOT / "dataset")
    results = {}
    if not skip_baseline:
        results["baseline"] = run_baseline()
    train("configs/resnet18.yaml")
    results["resnet18"] = run_evaluation("configs/resnet18.yaml", run_robustness=True,
                                         device_name="cpu")
    generate_gradcam_gallery(device_name="cpu")
    if ablation:
        train("configs/resnet18_no_color.yaml")
        results["resnet18_no_color"] = run_evaluation("configs/resnet18_no_color.yaml",
                                                      device_name="cpu")
    save_json({"runtime_minutes": (time.perf_counter() - started) / 60, "results": results},
              "outputs/metrics/all_experiments_summary.json")
    print("Finished. Only completed runs are included in the summary.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--ablation", action="store_true",
                        help="Also train/evaluate the optional no-color experiment.")
    args = parser.parse_args()
    run_all(args.skip_baseline, args.ablation)
