import numpy as np
import sounddevice as sd
import pyglet
import threading

# Audio config (inspired by karaoke_game/audio_sample.py)
CHUNK_SIZE = 1024  # Number of audio frames per buffer
RATE = 44100  # Audio sampling rate (HZ)
CHANNELS = 1  # Mono audio

print("Available input devices:\n")
devices = sd.query_devices()

input_devices = []
for i, dev in enumerate(devices):
    if dev["max_input_channels"] > 0:
        print(f"{i}: {dev['name']}")
        input_devices.append(i)

input_device = int(input("\nSelect input device: "))


def audio_callback(indata, frames, time_info, status):
    global current_freq
    freq = detect_pitch(indata[:, 0].copy())
    with freq_lock:
        current_freq = freq


stream = sd.InputStream(
    device=input_device,
    channels=CHANNELS,
    samplerate=RATE,
    blocksize=CHUNK_SIZE,
    callback=audio_callback,
    latency="low",
)

# Song: note name, frequency, duration in seconds
SONG = [
    ("E4", 329.63, 1.0),
    ("E4", 329.63, 1.0),
    ("F4", 349.23, 1.0),
    ("G4", 392.00, 1.0),
    ("G4", 392.00, 1.0),
    ("F4", 349.23, 1.0),
    ("E4", 329.63, 1.0),
    ("D4", 293.66, 1.0),
    ("C4", 261.63, 1.0),
    ("C4", 261.63, 1.0),
    ("D4", 293.66, 1.0),
    ("E4", 329.63, 1.0),
    ("E4", 329.63, 1.5),
    ("D4", 293.66, 0.5),
    ("D4", 293.66, 2.0),
]
TOLERANCE = 40  # in Hz

# Shared state
current_freq = 0.0
freq_lock = threading.Lock()


def detect_pitch(data):
    data = data - np.mean(data)
    if np.max(np.abs(data)) < 0.01:
        return 0.0
    windowed = data * np.hanning(len(data))
    fft = np.abs(np.fft.rfft(windowed, n=len(data) * 4))
    freqs = np.fft.rfftfreq(len(data) * 4, 1.0 / RATE)
    mask = (freqs >= 80) & (freqs <= 1000)
    return float(freqs[mask][np.argmax(fft[mask])])


# Pyglet window
window = pyglet.window.Window(500, 340, caption="Karaoke — Ode to Joy")
batch = pyglet.graphics.Batch()

title = pyglet.text.Label(
    "ODE TO JOY",
    font_size=20,
    x=250,
    y=310,
    anchor_x="center",
    color=(255, 255, 255, 255),
    batch=batch,
)
target = pyglet.text.Label(
    "",
    font_size=48,
    x=250,
    y=190,
    anchor_x="center",
    color=(100, 200, 255, 255),
    batch=batch,
)
info = pyglet.text.Label(
    "",
    font_size=13,
    x=250,
    y=140,
    anchor_x="center",
    color=(200, 200, 200, 255),
    batch=batch,
)
heard = pyglet.text.Label(
    "",
    font_size=13,
    x=250,
    y=110,
    anchor_x="center",
    color=(180, 180, 180, 255),
    batch=batch,
)
score = pyglet.text.Label(
    "Score: 0",
    font_size=13,
    x=250,
    y=75,
    anchor_x="center",
    color=(255, 220, 80, 255),
    batch=batch,
)
status = pyglet.text.Label(
    "Get ready...",
    font_size=11,
    x=250,
    y=40,
    anchor_x="center",
    color=(150, 150, 150, 255),
    batch=batch,
)

# Game state
state = {
    "index": 0,
    "timer": 0.0,
    "score": 0,
    "hold": 0.0,
    "phase": "countdown",
    "countdown": 4.0,
}


def update(dt):
    global current_freq
    s = state

    if s["phase"] == "countdown":
        s["countdown"] -= dt
        target.text = str(max(1, int(s["countdown"]) + 1))
        info.text = "Ode to Joy — sing the notes shown!"
        if s["countdown"] <= 0.0:
            s["phase"] = "playing"
            s["timer"] = 0.0
        return

    if s["phase"] == "finished":
        target.text = "Done!"
        info.text = f"Final score: {s['score']}"
        status.text = "Press R to restart"
        return

    note_name, note_freq, note_dur = SONG[s["index"]]
    s["timer"] += dt

    with freq_lock:
        freq = current_freq

    on_pitch = freq > 0 and abs(freq - note_freq) < TOLERANCE
    if on_pitch:
        s["hold"] += dt

    target.text = note_name
    target.color = (80, 255, 120, 255) if on_pitch else (100, 200, 255, 255)
    info.text = f"Target: {note_freq:.0f} Hz"
    heard.text = f"You:    {freq:.0f} Hz" if freq > 0 else "You:    (no input detected)"
    status.text = "On pitch!" if on_pitch else ""

    if s["timer"] >= note_dur:
        pts = int(100 * min(s["hold"] / (note_dur * 0.5), 1.0))
        s["score"] += pts
        score.text = f"Score: {s['score']}"
        s["index"] += 1
        s["timer"] = 0.0
        s["hold"] = 0.0
        if s["index"] >= len(SONG):
            s["phase"] = "finished"


@window.event
def on_draw():
    window.clear()
    batch.draw()


@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.ESCAPE:
        pyglet.app.exit()
    if symbol == pyglet.window.key.R:
        state.update(
            index=0, timer=0.0, score=0, hold=0.0, phase="countdown", countdown=4.0
        )
        score.text = "Score: 0"


# Start
print("Starting microphone...")
stream.start()

pyglet.clock.schedule_interval(update, 1 / 30)
pyglet.app.run()
stream.stop()
