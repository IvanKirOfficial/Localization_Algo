"""Triggered far-field azimuth localization. Input: synchronous float PCM [N, M]."""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import numpy as np


class SRPPhat:
    def __init__(self, positions, fs, step=2.0, band=(300., 8000.), c=343.):
        self.pos = np.asarray(positions, dtype=float)
        if (self.pos.ndim != 2 or self.pos.shape[1] != 2 or
                len(self.pos) < 3 or not np.isfinite(self.pos).all() or
                np.linalg.matrix_rank(self.pos - self.pos.mean(0)) < 2):
            raise ValueError('Use >=3 non-collinear microphone positions [x,y] in metres')
        if fs <= 0 or c <= 0 or not 0 < step <= 30 or not 0 < band[0] < band[1] < fs/2:
            raise ValueError('Invalid fs, speed, grid or frequency band')
        self.fs, self.band = fs, band
        self.angles = np.arange(-180., 180., step)
        u = np.c_[np.cos(np.deg2rad(self.angles)), np.sin(np.deg2rad(self.angles))]
        self.pairs = [(i, j) for i in range(len(self.pos)) for j in range(i+1, len(self.pos))]
        # u points TO source: arrival t_i = -r_i dot u / c.
        self.lags = np.array([-((self.pos[i]-self.pos[j]) @ u.T)*fs/c
                              for i, j in self.pairs])

    def locate(self, audio):
        x = np.asarray(audio, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.pos) or len(x) < 16 or not np.isfinite(x).all():
            raise ValueError('Invalid audio window')
        n = len(x)
        if np.max(np.abs(self.lags)) >= n-1:
            raise ValueError('Analysis window too short for array aperture')
        nfft = 1 << (2*n-1).bit_length()
        spec = np.fft.rfft((x-x.mean(0))*np.hanning(n)[:, None], n=nfft, axis=0)
        freq = np.fft.rfftfreq(nfft, 1/self.fs)
        mask = (freq >= self.band[0]) & (freq <= self.band[1])
        score = np.zeros(len(self.angles))
        valid_pairs = 0
        for k, (i, j) in enumerate(self.pairs):
            cross = spec[:, i]*spec[:, j].conj()
            magnitude = np.abs(cross)
            peak = magnitude[mask].max(initial=0.)
            if peak < 1e-15:
                continue
            # Reject almost empty bins rather than amplify numerical noise.
            keep = mask & (magnitude > peak*1e-6)
            phat = np.zeros_like(cross)
            phat[keep] = cross[keep]/magnitude[keep]
            cc = np.fft.irfft(phat, nfft)
            lag = self.lags[k]
            lo = np.floor(lag).astype(int)
            frac = lag-lo
            score += (1-frac)*cc[lo % nfft] + frac*cc[(lo+1) % nfft]
            valid_pairs += 1
        if not valid_pairs:
            return {'azimuth_deg': None, 'spatial_contrast': 0., 'valid_pairs': 0}
        score /= valid_pairs
        best = int(score.argmax())
        distance = np.abs((self.angles-self.angles[best]+180) % 360-180)
        other = score[distance >= 15.]
        # Diagnostic only, not a calibrated probability.
        contrast = float((score[best]-other.max())/(np.std(score)+1e-12)) if len(other) else 0.
        return {'azimuth_deg': float(self.angles[best]),
                'spatial_contrast': contrast, 'valid_pairs': valid_pairs}


class TriggeredLocalizer:
    """Push consecutive 20 ms blocks. Times are relative to first input sample.

    Warm up on background; keep 160 ms pre-trigger, collect 240 ms post-trigger.
    No inference of sound class is made. Run SRP outside capture callback.
    """
    def __init__(self, positions, fs=24000, threshold_db=10., min_dbfs=-45.,
                 warmup_s=1., offset_s=0.):
        if not .1 <= warmup_s <= 3.:
            raise ValueError('Warmup must be between 0.1 and 3 s')
        self.srp = SRPPhat(positions, fs)
        self.fs, self.hop = fs, round(.02*fs)
        self.threshold, self.min_dbfs = threshold_db, min_dbfs
        self.warmup = int(np.ceil(warmup_s*fs/self.hop))
        self.offset = offset_s
        self.history = deque(maxlen=8)
        self.levels = deque(maxlen=150)
        self.index, self.cooldown_until = 0, 0
        self.pending = None

    def process(self, block):
        x = np.asarray(block, dtype=float)
        if x.shape != (self.hop, len(self.srp.pos)) or not np.isfinite(x).all():
            raise ValueError(f'Expected finite PCM block {(self.hop, len(self.srp.pos))}')
        x = x.copy()
        # Per-channel energy avoids cancellation from averaging waveforms.
        power = np.mean((x-x.mean(0))**2, axis=0)
        db = float(10*np.log10(np.median(power)+1e-12))
        base = float(np.median(self.levels)) if self.levels else db
        result = None
        if self.pending is not None:
            self.pending['blocks'].append(x)
            self.pending['remaining'] -= 1
            if self.pending['remaining'] == 0:
                result = self._finish(False)
        elif (len(self.levels) >= self.warmup and self.index >= self.cooldown_until
              and db >= self.min_dbfs and db-base >= self.threshold):
            self.pending = {'blocks': list(self.history)+[x], 'remaining': 11,
                            'trigger': self.index, 'start': self.index-len(self.history),
                            'rise_db': db-base}
            self.cooldown_until = self.index+40
        else:
            # Freeze background during pending event and refractory interval.
            if self.index >= self.cooldown_until:
                self.levels.append(db)
        self.history.append(x)
        self.index += 1
        return result

    def _finish(self, truncated):
        event = self.pending
        self.pending = None
        window = np.concatenate(event['blocks'])
        # Average short SRP maps by averaging independent window estimates is
        # intentionally avoided: select the most energetic 80 ms audio window.
        length = min(round(.08*self.fs), len(window))
        starts = range(0, len(window)-length+1, self.hop)
        start = max(starts, key=lambda s: np.mean(window[s:s+length]**2))
        result = self.srp.locate(window[start:start+length])
        result.update(trigger_time_s=self.offset+event['trigger']*self.hop/self.fs,
                      window_start_s=self.offset+event['start']*self.hop/self.fs,
                      analysis_time_s=self.offset+(event['start']*self.hop+start)/self.fs,
                      rise_db=round(event['rise_db'], 2), sound_class='unclassified',
                      clipped_fraction=float(np.mean(np.abs(window) >= .999)),
                      truncated=truncated)
        return result

    def flush(self):
        return self._finish(True) if self.pending is not None else None


def main():
    import wave
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wav', type=Path)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('events.jsonl'))
    parser.add_argument('--offset', type=float, default=0., help='WAV start on video timeline, seconds')
    args = parser.parse_args()
    with wave.open(str(args.wav), 'rb') as source, args.out.open('w', encoding='utf-8') as out:
        geometry = json.loads(args.geometry.read_text(encoding='utf-8'))
        engine = TriggeredLocalizer(geometry, source.getframerate(), offset_s=args.offset)
        if source.getsampwidth() != 2 or source.getcomptype() != 'NONE':
            raise ValueError('CLI accepts uncompressed PCM16 WAV; convert with FFmpeg')
        if source.getnchannels() != len(geometry):
            raise ValueError('Channel count and geometry differ; do not downmix')
        def emit(event):
            if event is not None:
                line = json.dumps(event, ensure_ascii=False, allow_nan=False)
                out.write(line+'\n')
                print(line)
        while True:
            raw = source.readframes(engine.hop)
            if not raw:
                break
            block = np.frombuffer(raw, dtype='<i2').reshape(-1, len(geometry)).astype(float)/32768.
            if len(block) < engine.hop:
                block = np.pad(block, ((0, engine.hop-len(block)), (0, 0)))
            emit(engine.process(block))
        emit(engine.flush())


if __name__ == '__main__':
    main()
