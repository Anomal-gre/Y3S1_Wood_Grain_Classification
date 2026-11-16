import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.make_submission_zip import make_zip


class SubmissionTests(unittest.TestCase):
    def test_package_includes_pdf_and_source_but_excludes_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = ["README.md", "requirements.txt", "report/main.pdf", "report/main.tex",
                     "report/refs.bib", "src/model.py", "configs/resnet18.yaml",
                     "dataset/raw/private.jpg", "tmp/tool.exe", ".venv/secret.py",
                     "outputs/checkpoints/model.pth", "hf_space_static/assets/model.onnx",
                     "report/main.aux"]
            for name in files:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            (root / "report/main.tex").write_text(r"\input{figures/panel.tex}")
            (root / "report/figures").mkdir()
            (root / "report/figures/panel.tex").write_text(r"\includegraphics{wood.png}")
            (root / "report/figures/wood.png").write_bytes(b"fixture")
            (root / "report/figures/unused.png").write_bytes(b"fixture")
            archive = make_zip(project_root=root)
            self.assertEqual(archive.name, "INT3401_11_Nhom18.zip")
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
            self.assertIn("report/main.pdf", names)
            self.assertIn("src/model.py", names)
            self.assertIn("report/figures/wood.png", names)
            self.assertNotIn("report/figures/unused.png", names)
            for name in files[7:]:
                self.assertNotIn(name, names)

    def test_missing_pdf_prevents_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report").mkdir()
            for name in ["README.md", "requirements.txt", "report/main.tex", "report/refs.bib"]:
                (root / name).write_text("fixture")
            with self.assertRaisesRegex(FileNotFoundError, "main.pdf"):
                make_zip(project_root=root)
