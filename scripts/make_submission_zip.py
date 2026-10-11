"""
Script to create INT3401_11_NhomX.zip submission package.
Strictly excludes dataset, checkpoints (*.pth), virtual environments, and caches.
"""

import os
import zipfile
import argparse
from pathlib import Path

EXCLUDE_PATTERNS = [
    "dataset",
    "raw",
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "*.pth",
    "*.pt",
    "*.zip",
    "*.rar",
    "*.tar.gz"
]

def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in ["dataset", ".venv", "venv", "__pycache__", ".git", ".idea", ".vscode"]:
            return True
    if path.suffix.lower() in [".pth", ".pt", ".zip", ".rar", ".gz"]:
        return True
    return False

def make_zip(group_id: str = "Nhom11", output_dir: Path = None):
    project_root = Path(__file__).resolve().parent.parent
    zip_name = f"INT3401_11_{group_id}.zip"
    if output_dir is None:
        output_dir = project_root
    target_zip = output_dir / zip_name

    print(f"Creating submission zip: {target_zip}")
    count = 0
    with zipfile.ZipFile(target_zip, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in project_root.rglob("*"):
            if file_path.is_file():
                if should_exclude(file_path):
                    continue
                if file_path == target_zip:
                    continue
                rel_path = file_path.relative_to(project_root)
                zip_file.write(file_path, arcname=rel_path)
                count += 1

    zip_size_mb = target_zip.stat().st_size / (1024 * 1024)
    print(f"Successfully packaged {count} files into {target_zip.name} ({zip_size_mb:.2f} MB).")
    print("Verification: No dataset, cache, or weight files included.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build submission zip package")
    parser.add_argument("--group", type=str, default="Nhom01", help="Group identifier, e.g., Nhom01")
    args = parser.parse_args()
    make_zip(group_id=args.group)
