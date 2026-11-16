"""Offline checks for inference geometry, CAM and head-only training."""
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from src.baseline_lbp_svm import extract_texture_features, run_baseline
from src.dataset import MEAN, STD, build_eval_transform, build_eval_image_transform
from src.gradcam import GradCAM, get_target_layer, overlay_cam
from src.inference import load_predictor, predict_image
from src.model import WoodNet
from src.train import run_epoch


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (160, 240, 3), dtype=np.uint8))

    def test_displayed_cam_view_matches_model_tensor(self):
        view = build_eval_image_transform(200)(self.image)
        expected = TF.normalize(TF.to_tensor(view), MEAN, STD)
        actual = build_eval_transform(200)(self.image)
        self.assertEqual(view.size, (200, 200))
        self.assertTrue(torch.equal(expected, actual))

    def test_missing_checkpoint_is_not_a_random_weight_demo(self):
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
            load_predictor(ckpt_path="outputs/checkpoints/definitely_missing.pth")

    def test_inference_and_cam_with_frozen_backbone(self):
        model = WoodNet(pretrained=False)
        model.freeze_backbone()
        cfg = {"data": {"img_size": 200}, "model": {"num_classes": 12}}
        results, tensor, view = predict_image(model, cfg, torch.device("cpu"), self.image)
        self.assertEqual(len(results), 3)
        self.assertGreaterEqual(results[0][1], results[1][1])
        self.assertLessEqual(sum(score for _, score in results), 1.000001)
        cam = GradCAM(model, get_target_layer(model))
        try:
            with torch.no_grad():
                heatmap = cam.generate(tensor, results[0][0]["id"] - 1)
            self.assertEqual(heatmap.shape, (7, 7))
            self.assertTrue(np.isfinite(heatmap).all())
            self.assertEqual(overlay_cam(view, heatmap).shape, (200, 200, 3))
        finally:
            cam.close()
        self.assertFalse(get_target_layer(model)._forward_hooks)

    def test_head_warmup_updates_head_without_updating_backbone_bn(self):
        model = WoodNet(pretrained=False)
        model.freeze_backbone()
        before_bn = model.backbone.bn1.running_mean.clone()
        before_head = model.head[1].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=.001)
        batch = [(torch.randn(2, 3, 200, 200), torch.tensor([0, 1]))]
        loss, acc, f1 = run_epoch(model, batch, torch.nn.CrossEntropyLoss(), optimizer,
                                  torch.device("cpu"), True, True)
        self.assertTrue(np.isfinite([loss, acc, f1]).all())
        self.assertTrue(torch.equal(before_bn, model.backbone.bn1.running_mean))
        self.assertFalse(torch.equal(before_head, model.head[1].weight))

    def test_texture_vector_has_64_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wood.jpg"
            self.image.save(path)
            features = extract_texture_features(str(path))
        self.assertEqual(features.shape, (64,))
        self.assertTrue(np.isfinite(features).all())

    def test_baseline_model_is_saved_and_reloadable(self):
        from joblib import load
        features = np.array([[0., 0.], [0., .1], [1., 1.], [1., .9]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "svm.joblib"
            with patch("src.baseline_lbp_svm.load_split_samples", return_value=([], {})), \
                 patch("src.baseline_lbp_svm.extract_features_dataset", return_value=(features, labels)):
                run_baseline(output_file=str(Path(directory) / "metrics.json"), model_file=str(model_path))
            np.testing.assert_array_equal(load(model_path).predict(features), labels)


if __name__ == "__main__":
    unittest.main()
