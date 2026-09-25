import numpy as np

def calculate_ratio(landmarks, indices):
    """
    Вспомогательная функция для расчета отношения расстояний (Aspect Ratio)
    по формуле евклидова расстояния между точками.
    """
    # Вытаскиваем координаты x и y для каждой из 6 ключевых точек
    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
    p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])
    
    # Считаем вертикальные и горизонтальные расстояния
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    
    # Возвращаем итоговый коэффициент
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def calculate_ear(landmarks):
    """Расчет среднего EAR для левого и правого глаза"""
    # Индексы точек MediaPipe Face Mesh для левого и правого глаза
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    left_ear = calculate_ratio(landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_ratio(landmarks, RIGHT_EYE_INDICES)
    
    return (left_ear + right_ear) / 2.0

def calculate_mar(landmarks):
    """Расчет MAR по внутреннему контуру губ"""
    MOUTH_INDICES = [78, 81, 311, 308, 402, 178]
    return calculate_ratio(landmarks, MOUTH_INDICES)

def calculate_head_pose(matrix):
    """
    Расчет углов наклона и поворота головы (Pitch, Yaw, Roll) в градусах
    на основе 4x4 матрицы трансформации лица в метрическом 3D-пространстве (metric 3D space).
    
    :param matrix: 4x4 numpy-массив трансформации из MediaPipe FaceLandmarker
    :return: кортеж (pitch, yaw, roll) в градусах или (None, None, None)
    
    Ориентация углов:
    - Pitch: наклон вверх/вниз (+ кивок вниз / "клевание носом", - запрокидывание назад)
    - Yaw: поворот влево/вправо (+ поворот направо, - поворот налево)
    - Roll: наклон к плечу (+ наклон к правому плечу, - к левому плечу)
    """
    if matrix is None:
        return None, None, None

    import cv2
    # Извлекаем 3x3 матрицу вращения
    rotation_matrix = matrix[:3, :3]
    
    # Декомпозиция матрицы вращения на углы Эйлера
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    
    return float(pitch), float(yaw), float(roll)

def draw_head_pose_axes(frame, matrix, landmarks, axis_length=None):
    """
    Отрисовывает 3D оси координат ориентации головы на кадре, начиная от кончика носа:
    - Красная ось (X): вправо
    - Зеленая ось (Y): вверх
    - Синяя ось (Z): вперед (направление взгляда/лица)
    Масштаб осей и толщина линий автоматически адаптируются под разрешение кадра.
    """
    if matrix is None or landmarks is None:
        return frame

    import cv2
    h, w, _ = frame.shape
    scale = min(w / 640.0, h / 480.0)

    if axis_length is None:
        axis_len = int(55 * scale)
    elif axis_length in (50, 60):
        axis_len = int(axis_length * scale)
    else:
        axis_len = int(axis_length)

    # Индекс 1 в MediaPipe FaceMesh — кончик носа
    nose = landmarks[1]
    cx, cy = int(nose.x * w), int(nose.y * h)

    R = matrix[:3, :3]

    # Проекция единичных векторов канонической системы в 2D координаты изображения
    # В экранных координатах ось Y направлена вниз, поэтому инвертируем знак Y
    p_x = (int(cx + R[0, 0] * axis_len), int(cy - R[1, 0] * axis_len))
    p_y = (int(cx + R[0, 1] * axis_len), int(cy - R[1, 1] * axis_len))
    p_z = (int(cx + R[0, 2] * axis_len), int(cy - R[1, 2] * axis_len))

    line_thickness = max(1, int(round(2 * scale)))
    nose_radius = max(2, int(round(3 * scale)))

    # Отрисовка стрелок для осей: X - красный, Y - зеленый, Z - синий
    cv2.arrowedLine(frame, (cx, cy), p_x, (0, 0, 255), line_thickness, tipLength=0.2)
    cv2.arrowedLine(frame, (cx, cy), p_y, (0, 255, 0), line_thickness, tipLength=0.2)
    cv2.arrowedLine(frame, (cx, cy), p_z, (255, 0, 0), line_thickness, tipLength=0.2)
    cv2.circle(frame, (cx, cy), nose_radius, (0, 255, 255), -1)

    return frame


