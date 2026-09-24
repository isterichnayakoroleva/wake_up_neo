import cv2
import time
from src.detection import FaceDetector
from src.metrics import calculate_ear, calculate_mar, calculate_head_pose, draw_head_pose_axes
from src.alerts import DrowsinessAlertSystem

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 81, 311, 308, 402, 178]

# рисуем точки трекинга

def lets_draw(frame, landmarks, indices, color=(0, 255, 255)):
    h, w, _ = frame.shape

    for idx in indices:
        point = landmarks[idx]
        cx, cy = int(point.x * w), int(point.y * h)
        cv2.circle(frame, (cx, cy), 3, color, -1)

def main():
    # Инициализируем видеопоток с камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть веб-камеру.")
        return

    # Создаем экземпляр нашей системы предупреждений
    alert_system = DrowsinessAlertSystem(
        ear_threshold=0.20,
        mar_threshold=0.50,
        consecutive_frames=15,
        pitch_down_threshold=18.0,
        pitch_up_threshold=-20.0,
        roll_threshold=20.0,
        yaw_threshold=30.0,
        head_consecutive_frames=15
    )
    
    # Для MediaPipe VIDEO mode нужен таймстемп в миллисекундах
    start_time = time.time()

    print("Запуск системы мониторинга... Нажмите 'q' для выхода.")

    # Используем контекстный менеджер для FaceDetector
    with FaceDetector() as detector:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Не удалось получить кадр с камеры.")
                break

            # Отзеркалим кадр для удобства восприятия
            frame = cv2.flip(frame, 1)
            
            # Вычисляем текущий таймстемп
            timestamp_ms = int((time.time() - start_time) * 1000)

            # 1. Получаем ключевые точки лица и 3D метрическую матрицу трансформации
            landmarks, matrix = detector.get_face_data(frame, timestamp_ms)

            if landmarks:

                # рисуем точки
                lets_draw(frame, landmarks, LEFT_EYE_INDICES, color=(255, 255, 0))
                lets_draw(frame, landmarks, RIGHT_EYE_INDICES, color=(255, 255, 0))
                lets_draw(frame, landmarks, MOUTH_INDICES, color=(0, 0, 255))

                # 2. Считаем коэффициенты EAR и MAR
                ear = calculate_ear(landmarks)
                mar = calculate_mar(landmarks)

                # 3. Расчет ориентации и наклонов головы из metric 3D space
                pitch, yaw, roll = calculate_head_pose(matrix)

                # Отрисовка 3D-осей головы (красный: X, зеленый: Y, синий: Z)
                draw_head_pose_axes(frame, matrix, landmarks, axis_length=50)

                # 4. Передаем метрики в систему алертов
                alerts = alert_system.process_metrics(ear, mar, pitch, yaw, roll)

                # 5. Визуализация числовых показателей на экране
                cv2.putText(frame, f"EAR: {ear:.2f}", (30, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (30, 65), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                if pitch is not None:
                    # Цветовая индикация для наклона головы
                    pose_color = (0, 255, 0)
                    if alerts["sleep_danger_alert"] or alerts["distraction_alert"]:
                        pose_color = (0, 0, 255)
                    
                    cv2.putText(frame, f"Pitch (nod):  {pitch:+.1f} deg", (30, 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, pose_color, 2)
                    cv2.putText(frame, f"Yaw (turn):   {yaw:+.1f} deg", (30, 125),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, pose_color, 2)
                    cv2.putText(frame, f"Roll (tilt):  {roll:+.1f} deg", (30, 155),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, pose_color, 2)

                # Статистика за скользящее окно 60 секунд
                cv2.putText(frame, f"Window (60s) - Yawns: {alerts['yawns_in_window']} | Microsleeps: {alerts['microsleeps_in_window']}", 
                            (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                # 6. Раздельные статусы тревоги
                # А) Критическая опасность: Сон (падение EAR и/или кивок Pitch вниз)
                if alerts["sleep_danger_alert"]:
                    cv2.putText(frame, "ОПАСНОСТЬ: СОН!", (100, 230), 
                                cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)
                
                # Б) Отвлечение: Внимание на дорогу (критический Pitch/Yaw в сторону / боковые зеркала)
                if alerts["distraction_alert"]:
                    cv2.putText(frame, "ВНИМАНИЕ НА ДОРОГУ!", (70, 275), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 140, 255), 3)

                # В) Кумулятивная усталость (накопление зевков и микрозасыпаний за 60 секунд)
                if alerts["cumulative_fatigue_alert"]:
                    cv2.putText(frame, "УСТАЛОСТЬ: НУЖЕН ОТДЫХ", (70, 320), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.85, (0, 165, 255), 2)

                # Г) Зевок в реальном времени
                if alerts["yawn_alert"]:
                    cv2.putText(frame, "НЕ ЗЕВАТЬ", (160, 360), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.85, (0, 215, 255), 2)

            # Показываем итоговое окно пользователю
            cv2.imshow('wake up Neo', frame)

            # выход из программы
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Освобождаем ресурсы камеры и закрываем окна
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()