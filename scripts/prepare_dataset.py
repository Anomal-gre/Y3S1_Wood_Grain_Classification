"""
Prepare dataset splits for Wood Grain Classification.
Verifies images, builds stratified Train/Val splits from Train set,
and establishes the independent Test set. Generates dataset/splits.json.
"""

import json
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.species import SPECIES_MAPPING


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False

def prepare_splits(data_root: Path, seed: int = 42, val_ratio: float = 0.2):
    raw_dir = data_root / "raw" / "Wood Dataset"
    train_dir = raw_dir / "Train"
    test_dir = raw_dir / "Test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Raw dataset directories not found in {raw_dir}")

    manifest_path = PROJECT_ROOT / "configs/split_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if seed == manifest["seed"] and val_ratio == manifest["val_ratio"]:
            parts = {}
            for split in ("train", "val", "test"):
                parts[split] = []
                for relative in manifest[split]:
                    path = data_root.parent / relative
                    key = path.parent.name if split != "test" else path.stem.rsplit("_", 1)[1]
                    if not verify_image(path):
                        raise ValueError(f"Missing or damaged manifest image: {path}")
                    parts[split].append({"path": relative, "label": int(key) - 1, "class_key": key})
            counts = {key: len(value) for key, value in parts.items()}
            counts["total"] = sum(counts.values())
            result = {"metadata": {"dataset_name": "Wood Species Dataset (WOOD-AUTH)",
                                   "num_classes": 12, "seed": seed, "val_ratio": val_ratio,
                                   "counts": counts, "classes": SPECIES_MAPPING}, **parts}
            data_root.mkdir(parents=True, exist_ok=True)
            (data_root / "splits.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Restored recorded split from {manifest_path}: {counts}")
            return

    print("Verifying Train dataset images...")
    train_samples = []
    for cls_dir in sorted(train_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        if not cls_dir.is_dir():
            continue
        cls_key = cls_dir.name
        label_idx = int(cls_key) - 1  # 0-indexed: 0..11
        for img_path in sorted(cls_dir.glob("*.jpg")):
            if verify_image(img_path):
                # Use relative path from project root
                rel_path = img_path.relative_to(data_root.parent).as_posix()
                train_samples.append({"path": rel_path, "label": label_idx, "class_key": cls_key})
            else:
                print(f"[Warning] Corrupted image skipped: {img_path}")

    print("Verifying Test dataset images...")
    test_samples = []
    for img_path in sorted(test_dir.glob("*.jpg")):
        parts = img_path.stem.split("_")
        if len(parts) == 2 and parts[1] in SPECIES_MAPPING:
            cls_key = parts[1]
            label_idx = int(cls_key) - 1
            if verify_image(img_path):
                rel_path = img_path.relative_to(data_root.parent).as_posix()
                test_samples.append({"path": rel_path, "label": label_idx, "class_key": cls_key})
            else:
                print(f"[Warning] Corrupted image skipped: {img_path}")

    print(f"Total valid Train pool images: {len(train_samples)}")
    print(f"Total valid Test images: {len(test_samples)}")

    # Split train pool into stratified Train and Val
    train_labels = [s["label"] for s in train_samples]
    train_idx, val_idx = train_test_split(
        list(range(len(train_samples))),
        test_size=val_ratio,
        stratify=train_labels,
        random_state=seed
    )

    split_train = [train_samples[i] for i in train_idx]
    split_val = [train_samples[i] for i in val_idx]
    split_test = test_samples

    splits_data = {
        "metadata": {
            "dataset_name": "Wood Species Dataset (WOOD-AUTH)",
            "num_classes": len(SPECIES_MAPPING),
            "seed": seed,
            "val_ratio": val_ratio,
            "counts": {
                "train": len(split_train),
                "val": len(split_val),
                "test": len(split_test),
                "total": len(split_train) + len(split_val) + len(split_test)
            },
            "classes": SPECIES_MAPPING
        },
        "train": split_train,
        "val": split_val,
        "test": split_test
    }

    out_file = data_root / "splits.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully generated {out_file}:")
    print(f" - Train: {len(split_train)} images")
    print(f" - Val:   {len(split_val)} images")
    print(f" - Test:  {len(split_test)} images")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    prepare_splits(project_root / "dataset")
