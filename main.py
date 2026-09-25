import cv2
import time
from src.detection import FaceDetector
from src.metrics import calculate_ear, calculate_mar, calculate_head_pose, draw_head_pose_axes
from src.alerts import DrowsinessAlertSystem
from src.fatigue_decision import FatigueDecisionMaker, draw_fatigue_binary_indicator

LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 81, 311, 308, 402, 178]


def lets_draw(frame, landmarks, indices, color=(0, 255, 255)):
    """Отрисовка ключевых точек с адаптивным радиусом под любое разрешение кадра."""
    h, w, _ = frame.shape
    scale = min(w / 640.0, h / 480.0)
    radius = max(2, int(round(3 * scale)))

    for idx in indices:
        point = landmarks[idx]
        cx, cy = int(point.x * w), int(point.y * h)
        cv2.circle(frame, (cx, cy), radius, color, -1)


def draw_alert_banner(frame, text, center_y, scale=1.0, color=(0, 0, 255), base_font_scale=0.85, base_thickness=2):
    """Отрисовывает центрированный предупреждающий баннер с адаптивным размером и полупрозрачным фоном."""
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_COMPLEX
    f_scale = base_font_scale * scale
    th = max(2, int(round(base_thickness * scale)))
    (text_w, text_h), baseline = cv2.getTextSize(text, font, f_scale, th)

    cx = max(10, (w - text_w) // 2)
    cy = center_y

    pad_x = int(16 * scale)
    pad_y = int(8 * scale)

    overlay = frame.copy()
    cv2.rectangle(overlay, (cx - pad_x, cy - text_h - pad_y),
                  (cx + text_w + pad_x, cy + baseline + pad_y), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (cx - pad_x, cy - text_h - pad_y),
                  (cx + text_w + pad_x, cy + baseline + pad_y), color, max(1, int(round(2 * scale))))

    cv2.putText(frame, text, (cx, cy), font, f_scale, color, th)


def main():
    # Инициализируем видеопоток с камеры (разрешение может быть любым, графика масштабируется)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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

    # Модуль агрегации метрик усталости водителя (строгий бинарный вывод True/False с гистерезисом)
    fatigue_decision = FatigueDecisionMaker(
        fps=30.0,
        ear_threshold=0.20,
        pitch_down_threshold=18.0,
        mar_threshold=0.50,
        microsleep_duration=1.3,
        pitch_down_duration=1.5,
        combined_duration=0.8,
        alarm_hold_duration=2.0
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
            h, w, _ = frame.shape
            scale = min(w / 640.0, h / 480.0)

            # Вычисляем текущий таймстемп
            timestamp_ms = int((time.time() - start_time) * 1000)

            # 1. Получаем ключевые точки лица и 3D метрическую матрицу трансформации
            landmarks, matrix = detector.get_face_data(frame, timestamp_ms)

            if landmarks:
                # Рисуем точки с адаптивным радиусом
                lets_draw(frame, landmarks, LEFT_EYE_INDICES, color=(255, 255, 0))
                lets_draw(frame, landmarks, RIGHT_EYE_INDICES, color=(255, 255, 0))
                lets_draw(frame, landmarks, MOUTH_INDICES, color=(0, 0, 255))

                # 2. Считаем коэффициенты EAR и MAR
                ear = calculate_ear(landmarks)
                mar = calculate_mar(landmarks)

                # 3. Расчет ориентации и наклонов головы из metric 3D space
                pitch, yaw, roll = calculate_head_pose(matrix)

                # Отрисовка 3D-осей головы (красный: X, зеленый: Y, синий: Z) с авто-масштабом
                draw_head_pose_axes(frame, matrix, landmarks)

                # 4. Агрегация метрик усталости в единый бинарный флаг (True - спит, False - бодрствует)
                is_sleeping = fatigue_decision.process(ear, mar, pitch)

                # Передаем метрики в систему алертов
                alerts = alert_system.process_metrics(ear, mar, pitch, yaw, roll)

                # 5. Визуализация числовых показателей в полупрозрачной HUD-панели
                hud_x = int(15 * scale)
                hud_y = int(15 * scale)
                hud_w = int(360 * scale)
                hud_h = int(185 * scale)

                hud_overlay = frame.copy()
                cv2.rectangle(hud_overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (20, 25, 25), -1)
                cv2.addWeighted(hud_overlay, 0.60, frame, 0.40, 0, frame)
                cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (60, 75, 70), max(1, int(round(1 * scale))))

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.58 * scale
                font_scale_sm = 0.46 * scale
                text_thick = max(1, int(round(1.8 * scale)))
                text_thick_sm = max(1, int(round(1 * scale)))

                tx = hud_x + int(14 * scale)
                ty = hud_y + int(26 * scale)
                line_step = int(26 * scale)

                cv2.putText(frame, f"EAR: {ear:.2f}", (tx, ty), font, font_scale, (0, 255, 0), text_thick)
                cv2.putText(frame, f"MAR: {mar:.2f}", (tx, ty + line_step), font, font_scale, (0, 255, 0), text_thick)

                if pitch is not None:
                    # Цветовая индикация для наклона головы
                    pose_color = (0, 255, 0)
                    if alerts["sleep_danger_alert"] or alerts["distraction_alert"] or is_sleeping:
                        pose_color = (0, 0, 255)

                    cv2.putText(frame, f"Pitch (nod):  {pitch:+.1f} deg", (tx, ty + line_step * 2),
                                font, font_scale, pose_color, text_thick)
                    cv2.putText(frame, f"Yaw (turn):   {yaw:+.1f} deg", (tx, ty + line_step * 3),
                                font, font_scale, pose_color, text_thick)
                    cv2.putText(frame, f"Roll (tilt):  {roll:+.1f} deg", (tx, ty + line_step * 4),
                                font, font_scale, pose_color, text_thick)

                # Статистика за скользящее окно 60 секунд
                cv2.putText(frame, f"Window (60s) - Yawns: {alerts['yawns_in_window']} | Microsleeps: {alerts['microsleeps_in_window']}",
                            (tx, ty + line_step * 5 + int(2 * scale)), font, font_scale_sm, (220, 220, 220), text_thick_sm)

                # 6. Раздельные статусы тревоги (центрированные адаптивные баннеры)
                alert_y = int(h * 0.58)
                alert_step = int(48 * scale)

                # А) Критическая опасность: Сон (падение EAR и/или кивок Pitch вниз или взведенный флаг усталости)
                if alerts["sleep_danger_alert"] or is_sleeping:
                    draw_alert_banner(frame, "ОПАСНОСТЬ: СОН!", alert_y, scale=scale, color=(0, 0, 255), base_font_scale=0.95, base_thickness=2)
                    alert_y += alert_step

                # Б) Отвлечение: Внимание на дорогу (критический Pitch/Yaw в сторону / боковые зеркала)
                if alerts["distraction_alert"]:
                    draw_alert_banner(frame, "ВНИМАНИЕ НА ДОРОГУ!", alert_y, scale=scale, color=(0, 140, 255), base_font_scale=0.85, base_thickness=2)
                    alert_y += alert_step

                # В) Кумулятивная усталость (накопление зевков и микрозасыпаний за 60 секунд)
                if alerts["cumulative_fatigue_alert"]:
                    draw_alert_banner(frame, "УСТАЛОСТЬ: НУЖЕН ОТДЫХ", alert_y, scale=scale, color=(0, 165, 255), base_font_scale=0.80, base_thickness=2)
                    alert_y += alert_step

                # Г) Зевок в реальном времени
                if alerts["yawn_alert"]:
                    draw_alert_banner(frame, "НЕ ЗЕВАТЬ", alert_y, scale=scale, color=(0, 215, 255), base_font_scale=0.80, base_thickness=2)
            else:
                # Если лицо потеряно из кадра, обновляем состояние счетчиков
                is_sleeping = fatigue_decision.process(None, None, None)

            # 7. Отрисовка бинарного индикатора статуса засыпания (реле / сигнальная лампа)
            # Позиционирование в правом верхнем углу и размеры автоматически масштабируются под любое разрешение кадра
            draw_fatigue_binary_indicator(
                frame,
                is_sleeping,
                diagnostics=fatigue_decision.get_diagnostics()
            )

            # Показываем итоговое окно пользователю
            cv2.imshow('wake up Neo', frame)

            # Выход из программы
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Освобождаем ресурсы камеры и закрываем окна
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()