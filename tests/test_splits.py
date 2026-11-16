import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from scripts.prepare_dataset import prepare_splits


class SplitTests(unittest.TestCase):
    def test_manifest_preserves_recorded_order_and_class_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"seed": 42, "val_ratio": .2,
                        "train": ["dataset/raw/Wood Dataset/Train/2/z.jpg",
                                  "dataset/raw/Wood Dataset/Train/1/a.jpg"],
                        "val": ["dataset/raw/Wood Dataset/Train/1/b.jpg"],
                        "test": ["dataset/raw/Wood Dataset/Test/4_2.jpg"]}
            (root / "configs").mkdir()
            (root / "configs/split_manifest.json").write_text(json.dumps(manifest))
            for key in ["train", "val", "test"]:
                for relative in manifest[key]:
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8)).save(path)
            with patch("scripts.prepare_dataset.PROJECT_ROOT", root):
                prepare_splits(root / "dataset")
            saved = json.loads((root / "dataset/splits.json").read_text(encoding="utf-8"))
            self.assertEqual([x["path"] for x in saved["train"]], manifest["train"])
            self.assertEqual([x["label"] for x in saved["train"]], [1, 0])
            self.assertEqual(saved["test"][0]["label"], 1)
            self.assertEqual(saved["metadata"]["counts"]["total"], 4)
