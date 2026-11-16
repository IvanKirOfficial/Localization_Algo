import unittest
import numpy as np
from srp_phat import SRPPhat, TriggeredLocalizer

POS = np.array([[0, 0], [.06, 0], [.06, .06], [0, .06]])
FS = 24000


def wave(angle, seconds=.08, seed=42):
    rng = np.random.default_rng(seed)
    n = round(seconds*FS)
    base = rng.normal(0, .1, n+200)
    u = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    delay = -POS @ u/343*FS
    return np.stack([np.interp(np.arange(n)-d+100, np.arange(len(base)), base)
                     for d in delay], axis=1)


class Tests(unittest.TestCase):
    def test_quadrants_and_sign(self):
        for angle in [-150, -90, -30, 0, 45, 90, 140]:
            result = SRPPhat(POS, FS).locate(wave(angle))
            error = abs((result['azimuth_deg']-angle+180) % 360-180)
            self.assertLessEqual(error, 4, (angle, result))

    def test_silence(self):
        self.assertIsNone(SRPPhat(POS, FS).locate(np.zeros((1920, 4)))['azimuth_deg'])

    def test_stream_and_offset(self):
        rng = np.random.default_rng(7)
        engine = TriggeredLocalizer(POS, FS, offset_s=5.)
        x = rng.normal(0, .0001, (FS*3, 4))
        x[FS:FS+9600] += wave(45, .4)
        events = []
        for start in range(0, len(x), engine.hop):
            event = engine.process(x[start:start+engine.hop])
            if event:
                events.append(event)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]['trigger_time_s'], 6.)
        self.assertLessEqual(abs(events[0]['azimuth_deg']-45), 4)
        self.assertFalse(events[0]['truncated'])

    def test_no_background_events(self):
        engine = TriggeredLocalizer(POS)
        rng = np.random.default_rng(11)
        for _ in range(200):
            self.assertIsNone(engine.process(rng.normal(0, .001, (480, 4))))

    def test_flush_and_invalid(self):
        engine = TriggeredLocalizer(POS)
        for _ in range(60):
            engine.process(np.zeros((480, 4)))
        engine.process(wave(0, .02))
        self.assertTrue(engine.flush()['truncated'])
        with self.assertRaises(ValueError):
            engine.process(np.zeros((480, 1)))
        with self.assertRaises(ValueError):
            SRPPhat([[0, 0], [1, 0], [2, 0]], FS)


if __name__ == '__main__':
    unittest.main()
