import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.camera import UniversalCamera


class TestUniversalCamera(unittest.TestCase):

    def test_fallback_when_picamera2_missing(self):
        with patch.dict("sys.modules", {"picamera2": None}):
            cam = UniversalCamera(width=640, height=480, device_index=99999)
            # При несуществующем индексе устройства 99999 камера не открывается, но не падает с исключением
            self.assertFalse(cam.is_opened())
            self.assertFalse(cam.isOpened())
            success, frame = cam.read()
            self.assertFalse(success)
            self.assertIsNone(frame)
            cam.release()

    def test_mock_picamera2_backend(self):
        mock_picam2_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.capture_array.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_picam2_cls.return_value = mock_instance

        mock_module = MagicMock()
        mock_module.Picamera2 = mock_picam2_cls

        with patch.dict("sys.modules", {"picamera2": mock_module}):
            cam = UniversalCamera(width=640, height=480)
            self.assertEqual(cam.backend, "picam2")
            self.assertTrue(cam.is_opened())
            self.assertTrue(cam.isOpened())

            success, frame = cam.read()
            self.assertTrue(success)
            self.assertEqual(frame.shape, (480, 640, 3))

            cam.release()
            mock_instance.stop.assert_called_once()
            mock_instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
