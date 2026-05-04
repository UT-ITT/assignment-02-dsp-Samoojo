import numpy as np
import sounddevice as sd
from pynput.keyboard import Controller, Key
import collections
import time

# Audio config (inspired by karaoke_game/audio_sample.py)
CHUNK_SIZE = 1024  # Number of audio frames per buffer
RATE = 44100  # Audio sampling rate (HZ)
CHANNELS = 1  # Mono audio

# BandPass in Hz:
FREQ_MIN = 1000
FREQ_MAX = 4000
AMPLITUDE_THRESHOLD = 0.02  # minimum loudness

HISTORY_LEN = 12  # how many frames to track
CHIRP_SLOPE = 150  # Hz change to count as a chirp
COOLDOWN = 0.6  # seconds before another trigger

print("Available input devices:\n")
devices = sd.query_devices()

input_devices = []
for i, dev in enumerate(devices):
    if dev["max_input_channels"] > 0:
        print(f"{i}: {dev['name']}")
        input_devices.append(i)

input_device = int(input("\nSelect input device: "))

# State
freq_history = collections.deque(maxlen=HISTORY_LEN)
last_trigger = 0.0
keyboard = Controller()


# Return loudest frequency in whistle range
def detect_pitch(data):

    data = data - np.mean(data)
    if np.max(np.abs(data)) < AMPLITUDE_THRESHOLD:
        return 0.0
    windowed = data * np.hanning(len(data))
    fft = np.abs(np.fft.rfft(windowed, n=len(data) * 4))
    freqs = np.fft.rfftfreq(len(data) * 4, 1.0 / RATE)
    mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    if not np.any(mask):
        return 0.0
    peak = float(freqs[mask][np.argmax(fft[mask])])
    return peak


def audio_callback(indata, frames, time_info, status):
    global last_trigger

    freq = detect_pitch(indata[:, 0].copy())
    freq_history.append(freq)

    # check if history contains enough entries to count as chirp
    active = [f for f in freq_history if f > 0]
    if len(active) < HISTORY_LEN * 0.75:
        return

    # check slope
    x = np.arange(len(active), dtype=float)
    slope, _ = np.polyfit(x, active, 1)
    total_change = slope * len(active)

    now = time.time()
    if now - last_trigger < COOLDOWN:
        return

    if total_change > CHIRP_SLOPE:
        print(f"UP chirp detected  ({total_change:+.0f} Hz)")
        keyboard.press(Key.up)
        keyboard.release(Key.up)
        last_trigger = now
        freq_history.clear()

    elif total_change < -CHIRP_SLOPE:
        print(f"DOWN chirp detected  ({total_change:+.0f} Hz)")
        keyboard.press(Key.down)
        keyboard.release(Key.down)
        last_trigger = now
        freq_history.clear()


# Main
print("Whistle Input — chirp detector")
print("  Whistle low2high  →  UP")
print("  Whistle high2low  →  DOWN")
print("  Ctrl+C to quit\n")

with sd.InputStream(
    device=input_device,
    channels=1,
    samplerate=RATE,
    blocksize=CHUNK_SIZE,
    callback=audio_callback,
    latency="low",
):
    while True:
        time.sleep(0.1)
