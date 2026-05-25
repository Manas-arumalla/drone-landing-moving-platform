import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed")
class VisualServoingTests(unittest.TestCase):
    def test_detects_yellow_marker_error(self):
        import numpy as np

        from drone_landing.vision import detect_landing_marker

        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[:, :] = [20, 30, 35]
        image[35:45, 65:75] = [245, 210, 30]

        detection = detect_landing_marker(image)

        self.assertTrue(detection.visible)
        self.assertGreater(detection.error_x, 0.2)
        self.assertAlmostEqual(detection.error_y, 0.0, delta=0.2)

    def test_visual_controller_outputs_four_motor_commands(self):
        from drone_landing.vision import MarkerDetection, VisualServoController

        controller = VisualServoController()
        observation = [0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        action = controller.act(observation, MarkerDetection(visible=True, error_x=0.1, error_y=-0.2))

        self.assertEqual(len(action), 4)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in action))


if __name__ == "__main__":
    unittest.main()

