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
        
        self.blink_counter = 0
        self.nod_counter = 0
        self.combined_sleep_counter = 0
        self.distraction_counter = 0
        
        self.yawn_start_time = None
        self.is_yawning = False
        self.is_microsleeping = False

    def process_metrics(self, ear, mar, pitch=None, yaw=None, roll=None, current_time=None):

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

        is_looking_away = False
        if yaw is not None and abs(yaw) > self.yaw_threshold:
            is_looking_away = True
        if pitch is not None and pitch < self.pitch_up_threshold:
            is_looking_away = True

        if is_looking_away:
            self.distraction_counter += 1
            if self.distraction_counter >= self.head_consecutive_frames:
                status["distraction_alert"] = True
            
            self.blink_counter = 0
            self.nod_counter = 0
            self.combined_sleep_counter = 0
        else:
            self.distraction_counter = 0

            eyes_closed = (ear is not None and ear < self.ear_threshold)
            head_nodding = (pitch is not None and pitch > self.pitch_down_threshold)  # Кивок вниз ("клевание носом")
            head_tilted_side = (roll is not None and abs(roll) > self.roll_threshold)

            if eyes_closed and head_nodding:
                self.combined_sleep_counter += 1
            else:
                self.combined_sleep_counter = 0

            if eyes_closed:
                self.blink_counter += 1
            else:
                self.blink_counter = 0

            if head_nodding or head_tilted_side:
                self.nod_counter += 1
            else:
                self.nod_counter = 0

            if (self.combined_sleep_counter >= self.fast_sleep_frames or
                self.blink_counter >= self.consecutive_frames or
                self.nod_counter >= self.head_consecutive_frames):
                
                status["sleep_danger_alert"] = True
                status["drowsiness_alert"] = True
                
                if head_nodding:
                    status["head_drop_alert"] = True
                if head_tilted_side:
                    status["head_tilt_alert"] = True

                if not self.is_microsleeping:
                    self.microsleep_events.append(current_time)
                    self.is_microsleeping = True
            else:
                self.is_microsleeping = False

        if mar is not None and mar > self.mar_threshold:
            if not self.is_yawning:
                self.yawn_start_time = current_time
                self.is_yawning = True
            
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

        window_start = current_time - self.window_duration
        while self.yawn_events and self.yawn_events[0] < window_start:
            self.yawn_events.popleft()
        while self.microsleep_events and self.microsleep_events[0] < window_start:
            self.microsleep_events.popleft()

        status["yawns_in_window"] = len(self.yawn_events)
        status["microsleeps_in_window"] = len(self.microsleep_events)

        if (status["yawns_in_window"] >= self.max_yawns_in_window or 
            status["microsleeps_in_window"] >= self.max_microsleeps_in_window or
            (status["yawns_in_window"] >= 2 and status["microsleeps_in_window"] >= 1)):
            status["cumulative_fatigue_alert"] = True

        return status