import numpy as np

def calculate_ratio(landmarks, indices):
    p1 = np.array([landmarks[indices[0]].x, landmarks[indices[0]].y])
    p2 = np.array([landmarks[indices[1]].x, landmarks[indices[1]].y])
    p3 = np.array([landmarks[indices[2]].x, landmarks[indices[2]].y])
    p4 = np.array([landmarks[indices[3]].x, landmarks[indices[3]].y])
    p5 = np.array([landmarks[indices[4]].x, landmarks[indices[4]].y])
    p6 = np.array([landmarks[indices[5]].x, landmarks[indices[5]].y])
    
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def calculate_ear(landmarks):
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    left_ear = calculate_ratio(landmarks, LEFT_EYE_INDICES)
    right_ear = calculate_ratio(landmarks, RIGHT_EYE_INDICES)
    
    return (left_ear + right_ear) / 2.0

def calculate_mar(landmarks):
    MOUTH_INDICES = [78, 81, 311, 308, 402, 178]
    return calculate_ratio(landmarks, MOUTH_INDICES)

def calculate_head_pose(matrix):
    if matrix is None:
        return None, None, None

    import cv2
    rotation_matrix = matrix[:3, :3]
    
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = angles[0], angles[1], angles[2]
    
    return float(pitch), float(yaw), float(roll)

def draw_head_pose_axes(frame, matrix, landmarks, axis_length=None):
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

    nose = landmarks[1]
    cx, cy = int(nose.x * w), int(nose.y * h)

    R = matrix[:3, :3]

    p_x = (int(cx + R[0, 0] * axis_len), int(cy - R[1, 0] * axis_len))
    p_y = (int(cx + R[0, 1] * axis_len), int(cy - R[1, 1] * axis_len))
    p_z = (int(cx + R[0, 2] * axis_len), int(cy - R[1, 2] * axis_len))

    line_thickness = max(1, int(round(2 * scale)))
    nose_radius = max(2, int(round(3 * scale)))

    cv2.arrowedLine(frame, (cx, cy), p_x, (0, 0, 255), line_thickness, tipLength=0.2)
    cv2.arrowedLine(frame, (cx, cy), p_y, (0, 255, 0), line_thickness, tipLength=0.2)
    cv2.arrowedLine(frame, (cx, cy), p_z, (255, 0, 0), line_thickness, tipLength=0.2)
    cv2.circle(frame, (cx, cy), nose_radius, (0, 255, 255), -1)

    return frame


