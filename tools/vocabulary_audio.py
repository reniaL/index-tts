"""WAV preparation and sample-accurate vocabulary assembly (no model needed)."""

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_mono(path, sample_rate=None):
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("音频为空或含有无效采样值")
    audio = audio.mean(axis=1)
    if sample_rate is not None and rate != sample_rate:
        divisor = math.gcd(rate, sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, rate // divisor).astype(np.float32)
        rate = sample_rate
    return audio, rate


def trim_edges(audio, rate, margin_seconds=0.08):
    """Trim only edge silence; keep 80 ms beyond each detected speech boundary."""
    frame = max(1, round(rate * 0.01))
    padded = np.pad(audio, (0, (-len(audio)) % frame))
    rms = np.sqrt(np.mean(padded.reshape(-1, frame) ** 2, axis=1))
    threshold = max(1e-5, float(rms.max()) * 10 ** (-55 / 20))
    active = np.flatnonzero(rms > threshold)
    if not len(active):
        return audio.copy()
    margin = round(rate * margin_seconds)
    start = max(0, int(active[0]) * frame - margin)
    end = min(len(audio), (int(active[-1]) + 1) * frame + margin)
    return audio[start:end].copy()


def write_wav(path, audio, rate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.wav")
    sf.write(str(temp), np.clip(audio, -1, 1), rate, subtype="PCM_16")
    temp.replace(path)


def prepare_audio(source, destination, trim=False, sample_rate=None):
    audio, rate = read_mono(source, sample_rate)
    if trim:
        audio = trim_edges(audio, rate)
    write_wav(destination, audio, rate)
    return rate, len(audio)


def assemble_audio(paths, destination, repeats=2, repeat_gap=0.8, item_gap=1.5):
    if not paths:
        raise ValueError("没有可拼接的音频")
    if isinstance(repeats, bool) or int(repeats) != repeats or not 1 <= repeats <= 10:
        raise ValueError("重复次数必须为 1–10 的整数")
    if not all(math.isfinite(x) and 0 <= x <= 30 for x in (repeat_gap, item_gap)):
        raise ValueError("间隔必须在 0–30 秒之间")
    repeats = int(repeats)
    first, rate = read_mono(paths[0])
    repeat_samples, item_samples = round(rate * repeat_gap), round(rate * item_gap)
    repeat_silence = np.zeros(repeat_samples, dtype=np.float32)
    item_silence = np.zeros(item_samples, dtype=np.float32)
    # Stream to disk to keep memory bounded for long lists.
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp.wav")
    total = 0
    try:
        with sf.SoundFile(str(temp), mode="w", samplerate=rate, channels=1, subtype="PCM_16") as output:
            for index, path in enumerate(paths):
                audio = first if index == 0 else read_mono(path, rate)[0]
                for repetition in range(repeats):
                    output.write(np.clip(audio, -1, 1))
                    total += len(audio)
                    if repetition < repeats - 1:
                        output.write(repeat_silence)
                        total += repeat_samples
                if index < len(paths) - 1:
                    output.write(item_silence)
                    total += item_samples
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)
    return {"sample_rate": rate, "samples": total, "seconds": total / rate,
            "repeat_gap_samples": repeat_samples, "item_gap_samples": item_samples}
