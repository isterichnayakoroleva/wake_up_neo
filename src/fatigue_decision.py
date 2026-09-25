"""
Модуль агрегации метрик усталости водителя.
Класс FatigueDecisionMaker сводит покадровые признаки (EAR, MAR, Pitch)
к однозначному бинарному выводу (True / False), сигнализирующему о том, спит человек или нет.
"""

from typing import Optional, Dict, Any
import cv2


class FatigueDecisionMaker:
    """
    Класс для принятия решения об усталости и засыпании водителя на основе покадровых признаков.

    Требования и логика:
    1. Расчетный FPS видеопотока (по умолчанию 30).
    2. Временные счетчики:
       - Исключают ложные тревоги от естественного моргания (0.1–0.4 сек).
       - Исключают ложные тревоги от коротких взглядов на приборную панель (<1.5 сек).
    3. Пороги срабатывания:
       - Порог микросна: непрерывное удержание низкого EAR дольше 1.2–1.5 сек (по умолч. 1.3 с).
       - Порог критического наклона головы: Pitch вниз дольше 1.5 сек.
       - Комбинированное условие: быстрое закрытие глаз (>0.8 сек) одновременно с опусканием головы.
    4. Гистерезис / задержка сброса (Hold Delay):
       - После взведения флага True он удерживается активным минимум 1.5–2.0 сек (по умолч. 2.0 с),
         даже при первом открытии глаз, чтобы светодиод/реле не дребезжали и сигнал был устойчивым.
    5. Метод process(ear, mar, pitch) -> bool:
       - Возвращает строго булево значение:
         True  — водитель спит (опасность, включить сигнал/лампу/реле).
         False — норма (бодрствует, выключить сигнал/лампу).
    """

    def __init__(
        self,
        fps: float = 30.0,
        ear_threshold: float = 0.20,
        pitch_down_threshold: float = 18.0,
        mar_threshold: float = 0.50,
        microsleep_duration: float = 1.3,
        pitch_down_duration: float = 1.5,
        combined_duration: float = 0.8,
        alarm_hold_duration: float = 2.0,
    ):
        """
        :param fps: Расчетный FPS видеопотока (по умолчанию 30).
        :param ear_threshold: Порог закрытия глаз по EAR.
        :param pitch_down_threshold: Порог наклона головы вниз по Pitch (градусы, >0 кивок вниз).
        :param mar_threshold: Порог зевания по MAR.
        :param microsleep_duration: Порог микросна (длительность удержания низкого EAR, 1.2–1.5 с).
        :param pitch_down_duration: Порог наклона головы вниз (1.5 с).
        :param combined_duration: Порог быстрого закрытия глаз + опускания головы (0.8 с).
        :param alarm_hold_duration: Задержка сброса флага True (гистерезис, 1.5–2.0 с).
        """
        if fps <= 0:
            raise ValueError(f"FPS должен быть положительным числом, получено {fps}")

        self.fps = float(fps)
        self.ear_threshold = float(ear_threshold)
        self.pitch_down_threshold = float(pitch_down_threshold)
        self.mar_threshold = float(mar_threshold)

        self.microsleep_duration = float(microsleep_duration)
        self.pitch_down_duration = float(pitch_down_duration)
        self.combined_duration = float(combined_duration)
        self.alarm_hold_duration = float(alarm_hold_duration)

        # Конвертация секундных порогов в количество кадров
        self.microsleep_frames = max(1, int(round(self.microsleep_duration * self.fps)))
        self.pitch_down_frames = max(1, int(round(self.pitch_down_duration * self.fps)))
        self.combined_frames = max(1, int(round(self.combined_duration * self.fps)))
        self.alarm_hold_frames = max(1, int(round(self.alarm_hold_duration * self.fps)))

        # Внутренние непрерывные покадровые счетчики
        self.eye_closed_frames = 0
        self.pitch_down_frames_counter = 0
        self.combined_frames_counter = 0

        # Счетчик кадров удержания тревоги (гистерезис)
        self.hold_frames_remaining = 0

        # Результирующее бинарное состояние
        self.is_alarm_active: bool = False
        self.current_trigger_source: str = "none"

    def process(
        self,
        ear: Optional[float],
        mar: Optional[float],
        pitch: Optional[float]
    ) -> bool:
        """
        Обработка метрик текущего кадра.

        :param ear: Eye Aspect Ratio (коэффициент раскрытия глаз). None если лицо не распознано.
        :param mar: Mouth Aspect Ratio (коэффициент раскрытия рта).
        :param pitch: Угол наклона головы Pitch в градусах (>0 — кивок/наклон вниз).
        :return: Строго булево значение bool:
                 True  — водитель спит (опасность, включить сигнал/лампу/реле).
                 False — норма (бодрствует, выключить сигнал/лампу).
        """
        # 1. Проверяем признаки в текущем кадре
        eyes_closed = (ear is not None and ear < self.ear_threshold)
        head_down = (pitch is not None and pitch > self.pitch_down_threshold)

        # 2. Обновление счетчиков с фильтрацией ложных тревог:
        # Естественное моргание (0.1–0.4 с = 3–12 кадров) сбрасывается в 0 при открытии глаз,
        # не достигая порога микросна (1.2–1.5 с = 36–45 кадров).
        if eyes_closed:
            self.eye_closed_frames += 1
        else:
            self.eye_closed_frames = 0

        # Короткий взгляд на приборную панель (<1.5 с) сбрасывается при возврате взгляда на дорогу.
        if head_down:
            self.pitch_down_frames_counter += 1
        else:
            self.pitch_down_frames_counter = 0

        # Комбинированное условие: быстрое закрытие глаз одновременно с опусканием головы
        if eyes_closed and head_down:
            self.combined_frames_counter += 1
        else:
            self.combined_frames_counter = 0

        # 3. Проверка срабатывания условий засыпания в текущем кадре
        is_microsleep = self.eye_closed_frames >= self.microsleep_frames
        is_critical_pitch = self.pitch_down_frames_counter >= self.pitch_down_frames
        is_combined = self.combined_frames_counter >= self.combined_frames

        is_currently_triggered = is_microsleep or is_critical_pitch or is_combined

        # 4. Логика гистерезиса / задержки сброса (Hold Delay)
        if is_currently_triggered:
            # Тревога взведена: перезаряжаем таймер удержания
            self.is_alarm_active = True
            self.hold_frames_remaining = self.alarm_hold_frames

            if is_combined:
                self.current_trigger_source = "combined"
            elif is_microsleep:
                self.current_trigger_source = "microsleep"
            elif is_critical_pitch:
                self.current_trigger_source = "pitch_down"
        else:
            # Прямого триггера нет, удерживаем активное состояние заданное время
            if self.hold_frames_remaining > 0:
                self.hold_frames_remaining -= 1
                self.is_alarm_active = True
                self.current_trigger_source = "hysteresis_hold"
            else:
                self.is_alarm_active = False
                self.current_trigger_source = "none"

        return bool(self.is_alarm_active)

    def reset(self) -> None:
        """Сброс всех счетчиков и состояния тревоги."""
        self.eye_closed_frames = 0
        self.pitch_down_frames_counter = 0
        self.combined_frames_counter = 0
        self.hold_frames_remaining = 0
        self.is_alarm_active = False
        self.current_trigger_source = "none"

    def get_diagnostics(self) -> Dict[str, Any]:
        """Диагностические данные для телеметрии и отладки."""
        return {
            "is_sleeping": self.is_alarm_active,
            "trigger_source": self.current_trigger_source,
            "eye_closed_seconds": self.eye_closed_frames / self.fps,
            "pitch_down_seconds": self.pitch_down_frames_counter / self.fps,
            "combined_seconds": self.combined_frames_counter / self.fps,
            "hold_seconds_remaining": self.hold_frames_remaining / self.fps,
        }


def draw_fatigue_binary_indicator(
    frame,
    is_sleeping: bool,
    top_left: Optional[tuple] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    diagnostics: Optional[Dict[str, Any]] = None
):
    """
    Отрисовывает аккуратный светодиодный индикатор бинарного статуса засыпания на кадре OpenCV.
    Автоматически адаптирует размеры, шрифты и позицию под текущее разрешение кадра:
    - Зеленый индикатор: НОРМА (бодрствует, реле/лампа выключены).
    - Красный индикатор: СОН (опасность, реле/лампа включены).
    """
    h, w, _ = frame.shape
    scale = min(w / 640.0, h / 480.0)

    # Адаптивные размеры плашки
    if width is None:
        badge_w = int(225 * scale)
    else:
        badge_w = int(width * scale) if width <= 250 and scale > 1.0 else int(width)

    if height is None:
        badge_h = int(68 * scale)
    else:
        badge_h = int(height * scale) if height <= 80 and scale > 1.0 else int(height)

    # Автоматическое позиционирование в правом верхнем углу
    if top_left is None:
        margin_x = int(20 * scale)
        margin_y = int(18 * scale)
        x = w - badge_w - margin_x
        y = margin_y
    else:
        if top_left[0] > 350 and scale > 1.0:
            margin_x = int(20 * scale)
            margin_y = int(18 * scale)
            x = w - badge_w - margin_x
            y = margin_y
        else:
            x, y = int(top_left[0] * scale), int(top_left[1] * scale)

    overlay = frame.copy()

    if is_sleeping:
        bg_color = (0, 0, 160)
        border_color = (0, 0, 255)
        led_color = (0, 0, 255)
        status_text = "SLEEP ALERT!"
        flag_text = "RELAY: [ON]"
        
        # Адаптивная красная рамка вокруг всего кадра для тревоги
        frame_border_thick = max(2, int(round(5 * scale)))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), frame_border_thick)
    else:
        bg_color = (20, 40, 25)
        border_color = (0, 180, 0)
        led_color = (0, 255, 0)
        status_text = "NORMAL (AWAKE)"
        flag_text = "RELAY: [OFF]"

    # Полупрозрачная подложка плашки
    cv2.rectangle(overlay, (x, y), (x + badge_w, y + badge_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    badge_border_thick = max(1, int(round(2 * scale)))
    cv2.rectangle(frame, (x, y), (x + badge_w, y + badge_h), border_color, badge_border_thick)

    # Круглый светодиодный индикатор
    led_radius = max(4, int(round(10 * scale)))
    led_center = (x + int(24 * scale), y + badge_h // 2)
    cv2.circle(frame, led_center, led_radius, led_color, -1)
    led_halo_thick = max(1, int(round(1.5 * scale)))
    cv2.circle(frame, led_center, led_radius + max(1, int(round(2 * scale))), (255, 255, 255), led_halo_thick)

    # Текст статуса и реле
    font_scale_title = 0.52 * scale
    font_scale_sub = 0.44 * scale
    title_thick = max(1, int(round(2 * scale)))
    sub_thick = max(1, int(round(1 * scale)))

    text_x = x + int(44 * scale)
    title_y = y + int(28 * scale)
    sub_y = y + int(52 * scale)

    cv2.putText(frame, status_text, (text_x, title_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (255, 255, 255), title_thick)
    
    text_color = (130, 130, 255) if is_sleeping else (150, 255, 150)
    cv2.putText(frame, flag_text, (text_x, sub_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_sub, text_color, sub_thick)

    return frame
