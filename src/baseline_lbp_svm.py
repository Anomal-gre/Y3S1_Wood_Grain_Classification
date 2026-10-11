"""
Classical Baseline for Wood Grain Texture Classification:
Local Binary Patterns (LBP) + Gray-Level Co-occurrence Matrix (GLCM) + Support Vector Machine (SVM).
Experiment E1.
"""

import time
import json
import sys
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from src.utils import set_seed, save_json
from src.dataset import load_split_samples

def extract_texture_features(img_path: str) -> np.ndarray:
    """
    Extract multi-radius Local Binary Patterns (LBP) and GLCM statistics.
    Returns a compact texture feature vector.
    """
    with Image.open(img_path) as im:
        im_gray = np.array(im.convert("L"))

    features = []

    # 1. Multi-radius Uniform LBP Histograms (R=1, 2, 3 with P=8)
    for radius in [1, 2, 3]:
        n_points = 8 * radius
        lbp = local_binary_pattern(im_gray, n_points, radius, method="uniform")
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        features.extend(hist)

    # 2. GLCM features at 4 angles (0, 45, 90, 135 degrees)
    # Quantize to 32 levels for fast and robust GLCM computation
    quantized = (im_gray // 8).astype(np.uint8)
    glcm = graycomatrix(quantized, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=32, symmetric=True, normed=True)
    
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        vals = graycoprops(glcm, prop).ravel()
        features.extend([vals.mean(), vals.std()])

    return np.array(features, dtype=np.float32)

from joblib import Parallel, delayed

def extract_features_dataset(samples: List[Tuple[str, int]], desc: str = "Extracting") -> Tuple[np.ndarray, np.ndarray]:
    results = Parallel(n_jobs=-1, batch_size=32)(
        delayed(extract_texture_features)(path) for path, _ in tqdm(samples, desc=desc)
    )
    y = [label for _, label in samples]
    return np.array(results, dtype=np.float32), np.array(y, dtype=np.int64)

def run_baseline(
    splits_file: str = "dataset/splits.json",
    output_file: str = "outputs/metrics/baseline_lbp_svm.json",
    seed: int = 42
):
    set_seed(seed)
    print("=" * 60)
    print("Running Experiment E1: Classical Texture Baseline (LBP + GLCM + SVM)")
    print("=" * 60)

    # Load Train and Test splits
    train_samples, metadata = load_split_samples(splits_file, "train")
    test_samples, _ = load_split_samples(splits_file, "test")

    print(f"Train samples: {len(train_samples)}, Test samples: {len(test_samples)}")

    print("\n[Step 1/3] Extracting features for Train set...")
    X_train, y_train = extract_features_dataset(train_samples, desc="Train features")

    print("\n[Step 2/3] Extracting features for Test set...")
    X_test, y_test = extract_features_dataset(test_samples, desc="Test features")

    print("\n[Step 3/3] Training Support Vector Machine (RBF kernel)...")
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=seed))
    ])

    t_start_train = time.time()
    clf.fit(X_train, y_train)
    train_duration = time.time() - t_start_train
    print(f"Training completed in {train_duration:.2f}s.")

    # Evaluate inference latency
    t_start_inf = time.time()
    y_pred = clf.predict(X_test)
    inf_duration = time.time() - t_start_inf
    ms_per_image = (inf_duration / len(X_test)) * 1000

    test_acc = float(accuracy_score(y_test, y_pred))
    test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 40)
    print("BASELINE EVALUATION RESULTS (TEST SET):")
    print(f"Accuracy:        {test_acc * 100:.2f}%")
    print(f"Macro-F1:        {test_macro_f1 * 100:.2f}%")
    print(f"Inference speed: {ms_per_image:.2f} ms/image")
    print("=" * 40)
    print("\nDetailed Classification Report:")
    print(report_text)

    results = {
        "experiment": "E1_baseline_lbp_glcm_svm",
        "method": "Multi-radius LBP + GLCM + RBF SVM",
        "seed": seed,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_dim": int(X_train.shape[1]),
        "train_time_sec": round(train_duration, 2),
        "inference_latency_ms": round(ms_per_image, 2),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "classification_report": report_dict
    }

    save_json(results, output_file)
    print(f"\nSaved baseline metrics to {output_file}")
    return results

if __name__ == "__main__":
    run_baseline()
