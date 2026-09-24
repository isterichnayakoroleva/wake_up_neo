import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class FaceDetector:
    def __init__(self, model_path="models/face_landmarker.task"):
        """
        Инициализируем детектор лиц MediaPipe.
        Указываем путь к модели и режим работы (VIDEO).
        """
        # Настройки базовых опций и пути к файлу модели
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # Настройки для Face Landmarker. отслеживается только одно лицо
        self.options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO, 
            num_faces=1,                            
            output_face_blendshapes=False,          
            output_facial_transformation_matrixes=True
        )
        # Переменная для хранения самого объекта детектора
        self.detector = None

    def __enter__(self):
        """Позволяет использовать класс через конструкцию 'with' (контекстный менеджер)"""
        self.detector = vision.FaceLandmarker.create_from_options(self.options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Автоматически закрывает детектор при выходе из 'with' для очистки памяти"""
        if self.detector:
            self.detector.close()

    def get_landmarks(self, frame, timestamp_ms, return_matrix=False):
        """
        Принимает BGR кадр из OpenCV, конвертирует его и ищет ключевые точки лица.
        Если return_matrix=True, возвращает кортеж (landmarks, matrix).
        Иначе возвращает только landmarks для обратной совместимости.
        """
        landmarks, matrix = self.get_face_data(frame, timestamp_ms)
        if return_matrix:
            return landmarks, matrix
        return landmarks

    def get_face_data(self, frame, timestamp_ms):
        """
        Ищет ключевые точки лица и возвращает кортеж:
        (landmarks, transformation_matrix).
        matrix — это 4x4 матрица трансформации лица в метрическом 3D-пространстве (metric 3D space).
        Если лицо не обнаружено, возвращает (None, None).
        """
        # 1. MediaPipe Tasks требует кадры в формате RGB
        import cv2
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Создаем специальный объект изображения MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 3. Запускаем детекцию (передаем картинку и уникальный таймстемп кадра в мс)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        # 4. Если лицо найдено, возвращаем точки и матрицу трансформации первого лица
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
            return landmarks, matrix
        
        return None, None