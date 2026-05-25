"""Tests for B3: the learned CNN deck-pose detector (architecture + detector wiring, CPU).

Dataset generation (rendering) and GPU training are exercised separately; these tests validate the
network shape and the detector's back-projection wiring without GPU or rendering."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class DeckCNNTests(unittest.TestCase):
    def test_forward_shape_and_pixel_range(self):
        import torch

        from drone_landing.perception.cnn_detector import IN_CH, INPUT_HW, build_net

        net = build_net()()
        net.eval()
        out = net(torch.zeros(4, IN_CH, INPUT_HW, INPUT_HW))
        self.assertEqual(tuple(out.shape), (4, 3))
        self.assertTrue(bool((out[:, :2].abs() <= 1.0).all()))   # px, py squashed to [-1, 1]

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "opencv not installed")
    def test_detector_backprojects_to_rel_xy(self):
        import numpy as np

        from drone_landing.perception.camera import CameraModel
        from drone_landing.perception.cnn_detector import CNNDeckDetector

        cam = CameraModel(480, 480, 90.0)
        det = CNNDeckDetector(cam, weights_path=None, conf_thresh=0.0)   # untrained; force a detection
        img = (np.random.default_rng(0).random((480, 480, 3)) * 255).astype(np.uint8)
        d = det.detect(img, range_m=2.0)
        self.assertTrue(d.found)
        self.assertEqual(d.rel_xy.shape, (2,))
        self.assertTrue(np.all(np.isfinite(d.rel_xy)))

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "opencv not installed")
    def test_principal_point_maps_to_zero_rel_xy(self):
        # A predicted pixel at the camera principal point (cx, cy) must back-project to ~zero relative XY
        # (deck directly below). Normalised pixel for (cx,cy) is (2cx/W-1, 2cy/H-1).
        import numpy as np
        import torch

        from drone_landing.perception.camera import CameraModel
        from drone_landing.perception.cnn_detector import CNNDeckDetector

        cam = CameraModel(480, 480, 90.0)
        det = CNNDeckDetector(cam, weights_path=None, conf_thresh=0.0)
        px_n = 2 * cam.cx / cam.width - 1
        py_n = 2 * cam.cy / cam.height - 1
        det.net.forward = lambda t: torch.tensor([[px_n, py_n, 10.0]])
        d = det.detect(np.zeros((480, 480, 3), np.uint8), range_m=3.0)
        self.assertTrue(d.found)
        np.testing.assert_allclose(d.rel_xy, [0.0, 0.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
