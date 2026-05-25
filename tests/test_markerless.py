"""Synthetic unit test for the markerless deck tracker (no MuJoCo needed)."""

import importlib.util
import unittest

_HAS = importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2")


@unittest.skipUnless(_HAS, "numpy/opencv not installed")
class MarkerlessTests(unittest.TestCase):
    def test_backprojects_offset_blob(self):
        import numpy as np

        from drone_landing.perception import CameraModel
        from drone_landing.perception.markerless import MarkerlessDeckTracker

        cam = CameraModel(480, 480, 90.0)
        tracker = MarkerlessDeckTracker(cam)
        rng = 2.0
        # white pad blob centred at pixel (300, 180) on a black background
        img = np.zeros((480, 480, 3), dtype=np.uint8)
        cx_px, cy_px = 300, 180
        img[cy_px - 40:cy_px + 40, cx_px - 40:cx_px + 40] = 255
        det = tracker.detect(img, rng)
        self.assertTrue(det.found)
        # expected back-projection of the blob centre through the nadir pinhole
        exp_x = rng * (cx_px - cam.cx) / cam.fx
        exp_y = -rng * (cy_px - cam.cy) / cam.fy
        np.testing.assert_allclose(det.rel_xy, [exp_x, exp_y], atol=0.05)

    def test_rejects_when_no_bright_region(self):
        import numpy as np

        from drone_landing.perception import CameraModel
        from drone_landing.perception.markerless import MarkerlessDeckTracker

        tracker = MarkerlessDeckTracker(CameraModel(480, 480, 90.0))
        det = tracker.detect(np.zeros((480, 480, 3), dtype=np.uint8), 2.0)
        self.assertFalse(det.found)


if __name__ == "__main__":
    unittest.main()
