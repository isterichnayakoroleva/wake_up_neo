from typing import Tuple, Optional
import numpy as np
import cv2


class UniversalCamera:

    def __init__(self, width: int = 640, height: int = 480, device_index: int = 0):
        self.width = width
        self.height = height
        self.device_index = device_index
        self.backend: Optional[str] = None
        self.cam = None

        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "BGR888"}
            )
            picam2.configure(config)
            picam2.start()
            self.cam = picam2
            self.backend = "picam2"
            print("Успешно задействована камера Raspberry Pi CSI (Picamera2).")
            return
        except (ImportError, RuntimeError, Exception) as e:
            pass

        cap = cv2.VideoCapture(self.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if cap.isOpened():
            self.cam = cap
            self.backend = "cv2"
            print(f"Успешно задействована веб-камера через OpenCV VideoCapture (устройство {self.device_index}).")
        else:
            self.cam = None
            self.backend = None

    def is_opened(self) -> bool:
        if self.backend == "picam2":
            return self.cam is not None
        elif self.backend == "cv2":
            return self.cam is not None and self.cam.isOpened()
        return False

    def isOpened(self) -> bool:
        return self.is_opened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.backend == "picam2":
            try:
                frame = self.cam.capture_array()
                return True, frame
            except Exception:
                return False, None
        elif self.backend == "cv2":
            return self.cam.read()
        return False, None

    def release(self) -> None:
        if self.backend == "picam2" and self.cam is not None:
            try:
                self.cam.stop()
                self.cam.close()
            except Exception:
                pass
        elif self.backend == "cv2" and self.cam is not None:
            self.cam.release()
        self.cam = None
        self.backend = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
