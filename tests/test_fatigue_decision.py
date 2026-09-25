import unittest
import numpy as np
from src.fatigue_decision import FatigueDecisionMaker, draw_fatigue_binary_indicator


class TestFatigueDecisionMaker(unittest.TestCase):

    def test_initial_state_is_false(self):
        decision_maker = FatigueDecisionMaker(fps=30.0)
        self.assertFalse(decision_maker.is_alarm_active)
        res = decision_maker.process(ear=0.30, mar=0.20, pitch=0.0)
        self.assertFalse(res)
        self.assertIsInstance(res, bool)

    def test_natural_blinking_does_not_trigger_alarm(self):
        """
        Естественное моргание длится 0.1–0.4 сек (3–12 кадров при 30 FPS).
        Оно должно сбрасываться при открытии глаз и не вызывать ложной тревоги.
        """
        decision_maker = FatigueDecisionMaker(fps=30.0, microsleep_duration=1.3)

        for _ in range(10):
            for _ in range(9):
                res = decision_maker.process(ear=0.15, mar=0.20, pitch=0.0)
                self.assertFalse(res)

            for _ in range(30):
                res = decision_maker.process(ear=0.32, mar=0.20, pitch=0.0)
                self.assertFalse(res)

        self.assertFalse(decision_maker.is_alarm_active)

    def test_quick_dashboard_glances_do_not_trigger_alarm(self):
        """
        Короткий взгляд на приборную панель (0.5–0.8 сек) при открытых глазах
        не должен вызывать тревогу. Порог опускания головы 1.5 сек.
        """
        decision_maker = FatigueDecisionMaker(fps=30.0, pitch_down_duration=1.5)

        for _ in range(24):
            res = decision_maker.process(ear=0.30, mar=0.20, pitch=25.0)
            self.assertFalse(res)

        for _ in range(30):
            res = decision_maker.process(ear=0.30, mar=0.20, pitch=0.0)
            self.assertFalse(res)

        self.assertFalse(decision_maker.is_alarm_active)

    def test_microsleep_detection_and_hysteresis(self):
        """
        Микросон: непрерывное удержание низкого EAR >= 1.3 сек (39 кадров).
        После взведения True должен удерживаться минимум 2.0 сек (60 кадров).
        """
        decision_maker = FatigueDecisionMaker(
            fps=30.0,
            microsleep_duration=1.3,
            alarm_hold_duration=2.0
        )

        for frame_idx in range(38):
            res = decision_maker.process(ear=0.15, mar=0.20, pitch=0.0)
            self.assertFalse(res, f"Triggered too early at frame {frame_idx}")

        res = decision_maker.process(ear=0.15, mar=0.20, pitch=0.0)
        self.assertTrue(res)
        self.assertEqual(decision_maker.current_trigger_source, "microsleep")

        for hold_idx in range(60):
            res = decision_maker.process(ear=0.35, mar=0.20, pitch=0.0)
            self.assertTrue(res, f"Alarm dropped too early at hold frame {hold_idx}")
            self.assertEqual(decision_maker.current_trigger_source, "hysteresis_hold")

        res = decision_maker.process(ear=0.35, mar=0.20, pitch=0.0)
        self.assertFalse(res)
        self.assertEqual(decision_maker.current_trigger_source, "none")

    def test_critical_pitch_down(self):
        """
        Критический наклон головы (Pitch > 18.0) дольше 1.5 сек (45 кадров).
        """
        decision_maker = FatigueDecisionMaker(
            fps=30.0,
            pitch_down_duration=1.5,
            alarm_hold_duration=1.5
        )

        for _ in range(44):
            res = decision_maker.process(ear=0.30, mar=0.20, pitch=25.0)
            self.assertFalse(res)

        res = decision_maker.process(ear=0.30, mar=0.20, pitch=25.0)
        self.assertTrue(res)
        self.assertEqual(decision_maker.current_trigger_source, "pitch_down")

    def test_combined_sleep_condition(self):
        """
        Комбинированное условие: быстрое закрытие глаз (>0.8 сек = 24 кадра)
        одновременно с опусканием головы.
        """
        decision_maker = FatigueDecisionMaker(
            fps=30.0,
            combined_duration=0.8,
            microsleep_duration=1.3,
            pitch_down_duration=1.5
        )

        for _ in range(23):
            res = decision_maker.process(ear=0.15, mar=0.20, pitch=25.0)
            self.assertFalse(res)

        res = decision_maker.process(ear=0.15, mar=0.20, pitch=25.0)
        self.assertTrue(res)
        self.assertEqual(decision_maker.current_trigger_source, "combined")

    def test_handles_none_inputs(self):
        decision_maker = FatigueDecisionMaker(fps=30.0)
        res = decision_maker.process(ear=None, mar=None, pitch=None)
        self.assertFalse(res)
        self.assertIsInstance(res, bool)

    def test_reset_method(self):
        decision_maker = FatigueDecisionMaker(fps=30.0, microsleep_duration=1.0)
        for _ in range(35):
            decision_maker.process(ear=0.15, mar=0.20, pitch=0.0)

        self.assertTrue(decision_maker.is_alarm_active)
        decision_maker.reset()
        self.assertFalse(decision_maker.is_alarm_active)
        self.assertEqual(decision_maker.hold_frames_remaining, 0)
        self.assertEqual(decision_maker.eye_closed_frames, 0)

    def test_draw_fatigue_binary_indicator_resolutions(self):
        resolutions = [(480, 640), (720, 1280), (1080, 1920), (240, 320)]
        for h, w in resolutions:
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            res_false = draw_fatigue_binary_indicator(frame.copy(), is_sleeping=False)
            self.assertEqual(res_false.shape, (h, w, 3))
            res_true = draw_fatigue_binary_indicator(frame.copy(), is_sleeping=True)
            self.assertEqual(res_true.shape, (h, w, 3))


if __name__ == "__main__":
    unittest.main()
