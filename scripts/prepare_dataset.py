"""
Prepare dataset splits for Wood Grain Classification.
Verifies images, builds stratified Train/Val splits from Train set,
and establishes the independent Test set. Generates dataset/splits.json.
"""

import json
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SPECIES_MAPPING = {
    "1": {"id": 1, "code": "beech", "name": "Beech (Fagus sylvatica)", "vietnamese": "Dẻ gai"},
    "2": {"id": 2, "code": "black_locust", "name": "Black Locust (Robinia pseudoacacia)", "vietnamese": "Keo dậu hoa trắng"},
    "3": {"id": 3, "code": "cherry", "name": "Wild Cherry (Prunus avium)", "vietnamese": "Anh đào dại"},
    "4": {"id": 4, "code": "chestnut", "name": "Sweet Chestnut (Castanea sativa)", "vietnamese": "Dẻ Trùng Khánh"},
    "5": {"id": 5, "code": "douglas_fir", "name": "Douglas Fir (Pseudotsuga menziesii)", "vietnamese": "Thông Douglas"},
    "6": {"id": 6, "code": "european_larch", "name": "European Larch (Larix decidua)", "vietnamese": "Thông rụng lá Châu Âu"},
    "7": {"id": 7, "code": "oak", "name": "English Oak (Quercus robur)", "vietnamese": "Sồi châu Âu"},
    "8": {"id": 8, "code": "pine", "name": "Scots Pine (Pinus sylvestris)", "vietnamese": "Thông Scots"},
    "9": {"id": 9, "code": "oriental_plane", "name": "Oriental Plane (Platanus orientalis)", "vietnamese": "Ti trăn phương đông"},
    "10": {"id": 10, "code": "aspen", "name": "Aspen (Populus tremula)", "vietnamese": "Bạch dương rung"},
    "11": {"id": 11, "code": "tree_of_heaven", "name": "Tree of Heaven (Ailanthus altissima)", "vietnamese": "Cây thiên đường"},
    "12": {"id": 12, "code": "walnut", "name": "Walnut (Juglans regia)", "vietnamese": "Óc chó phương Tây"}
}

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

    print("Verifying Train dataset images...")
    train_samples = []
    for cls_dir in sorted(train_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        if not cls_dir.is_dir():
            continue
        cls_key = cls_dir.name
        label_idx = int(cls_key) - 1  # 0-indexed: 0..11
        for img_path in cls_dir.glob("*.jpg"):
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
