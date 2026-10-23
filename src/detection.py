import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    def __init__(self, model_path="models/face_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO, 
            num_faces=1,                            
            output_face_blendshapes=False,          
            output_facial_transformation_matrixes=True
        )
        self.detector = None

    def __enter__(self):
        self.detector = vision.FaceLandmarker.create_from_options(self.options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.detector:
            self.detector.close()

    def get_landmarks(self, frame, timestamp_ms, return_matrix=False):
        landmarks, matrix = self.get_face_data(frame, timestamp_ms)
        if return_matrix:
            return landmarks, matrix
        return landmarks

    def get_face_data(self, frame, timestamp_ms):
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
            return landmarks, matrix
        
        return None, None