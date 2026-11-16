"""Package the current report and source; never include local data or tools."""
import argparse
import re
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("src", "configs", "scripts", "tests", "docs")
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".txt"}

def submission_files(root):
    root = Path(root)
    required = ["README.md", "requirements.txt", "report/main.pdf", "report/main.tex", "report/refs.bib"]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing required submission file: {name}")
    files = {root / name for name in required}
    # Include the cover referenced by README in the offline submission.
    readme_cover = root / "docs/assets/readme-cover.png"
    if readme_cover.is_file():
        files.add(readme_cover)
    for directory in SOURCE_DIRS:
        for path in (root / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in SOURCE_SUFFIXES:
                files.add(path)
    # Follow actual LaTeX dependencies instead of bundling old, unused figures.
    pending = [root / "report/main.tex"]
    seen = set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        text = source.read_text(encoding="utf-8")
        for kind, name in re.findall(r"\\(input|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}", text):
            candidates = [root / "report" / name, root / "report/figures" / name]
            if kind == "input" and not Path(name).suffix:
                candidates = [p.with_suffix(".tex") for p in candidates]
            dependency = next((p for p in candidates if p.is_file()), None)
            if dependency is None:
                raise FileNotFoundError(f"Missing report dependency: {name}")
            dependency = dependency.resolve()
            if not dependency.is_relative_to((root / "report").resolve()):
                raise ValueError(f"Report dependency must be inside report/: {name}")
            files.add(dependency)
            if dependency.suffix == ".tex":
                pending.append(dependency)
    for pattern in ("*.json", "*_train.log"):
        files.update((root / "outputs/metrics").glob(pattern))
    for name in (".gitignore", ".gitattributes", "app.py"):
        if (root / name).is_file():
            files.add(root / name)
    return sorted(files)

def make_zip(group_id=18, output_dir=None, project_root=ROOT):
    root = Path(project_root).resolve()
    destination = Path(output_dir) if output_dir else root / "dist"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"INT3401_11_Nhom{int(group_id)}.zip"
    files = submission_files(root)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(target) as archive:
        if archive.testzip():
            raise RuntimeError("Archive verification failed")
        names = archive.namelist()
        assert "report/main.pdf" in names and "README.md" in names
    print(f"Created {target} ({target.stat().st_size / 1e6:.2f} MB, {len(files)} files)")
    return target

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", type=int, default=18, help="Official group number")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    make_zip(args.group, args.output_dir)
