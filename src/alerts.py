from collections import deque
import time

class DrowsinessAlertSystem:
    def __init__(self, 
                 ear_threshold=0.20, 
                 mar_threshold=0.50, 
                 consecutive_frames=15,
                 fast_sleep_frames=6,
                 pitch_down_threshold=18.0, 
                 pitch_up_threshold=-20.0,
                 roll_threshold=20.0, 
                 yaw_threshold=30.0, 
                 head_consecutive_frames=15,
                 window_duration=60.0,
                 max_yawns_in_window=3,
                 max_microsleeps_in_window=1):
        """
        Инициализация системы отслеживания усталости и внимания водителя.
        
        :param ear_threshold: Порог для закрытых глаз (если ниже — глаза закрыты)
        :param mar_threshold: Порог для зевка (если выше — рот открыт)
        :param consecutive_frames: Порог кадров для подтверждения закрытых глаз или наклона
        :param fast_sleep_frames: Ускоренный порог кадров при одновременном закрытии глаз и кивке вниз
        :param pitch_down_threshold: Угол наклона головы вниз для детекции "клевания носом" (градусы)
        :param pitch_up_threshold: Угол запрокидывания головы назад (градусы)
        :param roll_threshold: Угол наклона головы вбок к плечу (градусы)
        :param yaw_threshold: Угол поворота головы в сторону (зеркала, окно, градусы)
        :param head_consecutive_frames: Сколько кадров подряд голова в опасном положении для тревоги
        :param window_duration: Длительность скользящего окна в секундах (по умолчанию 60 сек)
        :param max_yawns_in_window: Допустимое кол-во зевков за окно до сигнала кумулятивной усталости
        :param max_microsleeps_in_window: Допустимое кол-во микрозасыпаний за окно
        """
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.consecutive_frames = consecutive_frames
        self.fast_sleep_frames = fast_sleep_frames
        
        self.pitch_down_threshold = pitch_down_threshold
        self.pitch_up_threshold = pitch_up_threshold
        self.roll_threshold = roll_threshold
        self.yaw_threshold = yaw_threshold
        self.head_consecutive_frames = head_consecutive_frames
        
        # Скользящее окно (таймстемпы событий)
        self.window_duration = window_duration
        self.max_yawns_in_window = max_yawns_in_window
        self.max_microsleeps_in_window = max_microsleeps_in_window
        
        self.yawn_events = deque()
        self.microsleep_events = deque()
        
        # Счетчики состояний
        self.blink_counter = 0
        self.nod_counter = 0
        self.combined_sleep_counter = 0
        self.distraction_counter = 0
        
        self.yawn_start_time = None
        self.is_yawning = False
        self.is_microsleeping = False

    def process_metrics(self, ear, mar, pitch=None, yaw=None, roll=None, current_time=None):
        """
        Принимает текущие значения EAR, MAR, и углы головы (Pitch, Yaw, Roll).
        Обновляет скользящее окно, счетчики и возвращает словарь со статусами тревоги:
        {
            "distraction_alert": True/False,        # «Внимание на дорогу» (боковое зеркало / поворот головы)
            "sleep_danger_alert": True/False,       # «Опасность: Сон» (падение EAR и/или кивок Pitch вниз)
            "yawn_alert": True/False,               # Зевок прямо сейчас
            "cumulative_fatigue_alert": True/False, # Критическая накопленная усталость за окно
            "yawns_in_window": int,                 # Кол-во зевков за последние 60 сек
            "microsleeps_in_window": int,           # Кол-во микрозасыпаний за последние 60 сек
            "drowsiness_alert": True/False,         # Алиас для sleep_danger_alert (обратная совместимость)
            "yawn_detected": True/False             # Алиас для yawn_alert (обратная совместимость)
        }
        """
        if current_time is None:
            current_time = time.time()

        status = {
            "distraction_alert": False,
            "sleep_danger_alert": False,
            "yawn_alert": False,
            "cumulative_fatigue_alert": False,
            "yawns_in_window": 0,
            "microsleeps_in_window": 0,
            "drowsiness_alert": False,
            "yawn_detected": False,
            "head_drop_alert": False,
            "head_tilt_alert": False
        }

        # -------------------------------------------------------------
        # 1. Проверка отвлечения: взгляд в боковое зеркало / в сторону
        # Водитель, смотрящий в зеркало, отвлекается, но НЕ спит!
        # -------------------------------------------------------------
        is_looking_away = False
        if yaw is not None and abs(yaw) > self.yaw_threshold:
            is_looking_away = True
        if pitch is not None and pitch < self.pitch_up_threshold:
            is_looking_away = True

        if is_looking_away:
            self.distraction_counter += 1
            if self.distraction_counter >= self.head_consecutive_frames:
                status["distraction_alert"] = True
            
            # Так как водитель смотрит в сторону/зеркало, сбрасываем счетчики сна,
            # чтобы искаженный ракурс глаза не вызывал ложной тревоги засыпания.
            self.blink_counter = 0
            self.nod_counter = 0
            self.combined_sleep_counter = 0
        else:
            self.distraction_counter = 0

            # ---------------------------------------------------------
            # 2. Проверка засыпания: «Опасность: Сон»
            # (Только если водитель НЕ смотрит в зеркало/сторону)
            # ---------------------------------------------------------
            eyes_closed = (ear is not None and ear < self.ear_threshold)
            head_nodding = (pitch is not None and pitch > self.pitch_down_threshold)  # Кивок вниз ("клевание носом")
            head_tilted_side = (roll is not None and abs(roll) > self.roll_threshold)

            # а) Синхронное падение EAR И кивок вниз (самый опасный сценарий — моментальная реакция)
            if eyes_closed and head_nodding:
                self.combined_sleep_counter += 1
            else:
                self.combined_sleep_counter = 0

            # б) Длительно закрытые глаза
            if eyes_closed:
                self.blink_counter += 1
            else:
                self.blink_counter = 0

            # в) Длительный кивок головы вниз или наклон вбок к плечу
            if head_nodding or head_tilted_side:
                self.nod_counter += 1
            else:
                self.nod_counter = 0

            # Триггер тревоги «Опасность: Сон»:
            # 1. Синхронно закрылись глаза и пошел кивок (ускоренное срабатывание fast_sleep_frames)
            # 2. Длительно закрыты глаза (consecutive_frames)
            # 3. Длительно опущенная голова (head_consecutive_frames)
            if (self.combined_sleep_counter >= self.fast_sleep_frames or
                self.blink_counter >= self.consecutive_frames or
                self.nod_counter >= self.head_consecutive_frames):
                
                status["sleep_danger_alert"] = True
                status["drowsiness_alert"] = True
                
                if head_nodding:
                    status["head_drop_alert"] = True
                if head_tilted_side:
                    status["head_tilt_alert"] = True

                # Регистрация эпизода микрозасыпания в скользящем окне
                if not self.is_microsleeping:
                    self.microsleep_events.append(current_time)
                    self.is_microsleeping = True
            else:
                self.is_microsleeping = False

        # -------------------------------------------------------------
        # 3. Проверка зевков (Рот)
        # -------------------------------------------------------------
        if mar is not None and mar > self.mar_threshold:
            if not self.is_yawning:
                self.yawn_start_time = current_time
                self.is_yawning = True
            
            # Зевок отображается активным, если длится более 1.5 секунд
            if (current_time - self.yawn_start_time) >= 1.5:
                status["yawn_alert"] = True
                status["yawn_detected"] = True
        else:
            if self.is_yawning:
                # Фиксируем завершенный зевок, если он длился достаточно долго
                if self.yawn_start_time and (current_time - self.yawn_start_time) >= 1.5:
                    self.yawn_events.append(current_time)
                self.is_yawning = False
                self.yawn_start_time = None

        # -------------------------------------------------------------
        # 4. Скользящее окно: очистка устаревших событий и расчет кумулятивной усталости
        # -------------------------------------------------------------
        window_start = current_time - self.window_duration
        while self.yawn_events and self.yawn_events[0] < window_start:
            self.yawn_events.popleft()
        while self.microsleep_events and self.microsleep_events[0] < window_start:
            self.microsleep_events.popleft()

        status["yawns_in_window"] = len(self.yawn_events)
        status["microsleeps_in_window"] = len(self.microsleep_events)

        # Кумулятивная усталость фиксируется при превышении лимита зевков или микроснов
        if (status["yawns_in_window"] >= self.max_yawns_in_window or 
            status["microsleeps_in_window"] >= self.max_microsleeps_in_window or
            (status["yawns_in_window"] >= 2 and status["microsleeps_in_window"] >= 1)):
            status["cumulative_fatigue_alert"] = True

        return status