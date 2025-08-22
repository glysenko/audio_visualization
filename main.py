import sys, importlib, threading, datetime
import numpy as np
import matplotlib
try:
    matplotlib.use("MacOSX")
except Exception:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SAMPLE_RATE      = 44100
BLOCK_SIZE       = 256
OSC_SECONDS      = 1.0
N_FFT            = 1024
MAX_FFT_HZ       = 5000
SPEC_HISTORY     = 100
PREFERRED_INPUT  = None
ANIM_INTERVAL_MS = 20
PNG_DPI          = 100

def ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

sd = importlib.import_module("sounddevice")

def pick_input_device(sd_mod, preferred=None):
    devices = sd_mod.query_devices()
    if isinstance(preferred, int):
        if 0 <= preferred < len(devices) and devices[preferred]["max_input_channels"] > 0:
            return preferred
    if isinstance(preferred, str):
        low = preferred.lower()
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0 and low in d["name"].lower():
                return i
    try:
        di = sd_mod.default.device[0] if isinstance(sd_mod.default.device,(list,tuple)) else sd_mod.default.device
        if di is not None and devices[di]["max_input_channels"]>0:
            return di
    except Exception:
        pass
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            return i
    return None

OSC_SAMPLES = int(round(OSC_SECONDS * SAMPLE_RATE))
ring        = np.zeros(OSC_SAMPLES, dtype=np.float32)
write_idx   = 0
idx_lock    = threading.Lock()

disp_time = np.zeros(OSC_SAMPLES, dtype=np.float32)
t_axis    = np.arange(OSC_SAMPLES, dtype=np.float32) / float(SAMPLE_RATE)

fft_time  = np.zeros(N_FFT, dtype=np.float32)
window    = np.hanning(N_FFT).astype(np.float32)
freqs     = np.fft.rfftfreq(N_FFT, d=1.0/float(SAMPLE_RATE))
spec_data = np.full((len(freqs), SPEC_HISTORY), -100.0, dtype=np.float32)

abs_time_s     = 0.0
state_lock     = threading.Lock()
paused         = False
paused_t_now   = None

def audio_callback(indata, frames, time_info, status):
    global write_idx, abs_time_s
    x = indata.mean(axis=1, dtype=np.float32) if indata.ndim==2 else indata.astype(np.float32, copy=False)
    n = x.shape[0]
    with idx_lock:
        wi = write_idx
        end = wi + n
        if end <= OSC_SAMPLES:
            ring[wi:end] = x
        else:
            k = OSC_SAMPLES - wi
            ring[wi:] = x[:k]
            ring[:end - OSC_SAMPLES] = x[k:]
        write_idx = end % OSC_SAMPLES
    with state_lock:
        abs_time_s += n / float(SAMPLE_RATE)

def start_stream():
    dev_idx = pick_input_device(sd, PREFERRED_INPUT)
    if dev_idx is None:
        raise RuntimeError("no input device")
    dev = sd.query_devices(dev_idx)
    in_ch = max(1, int(dev["max_input_channels"])) # каналы ввода
    st = sd.InputStream(device=dev_idx, channels=in_ch, samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE, dtype="float32",
                        callback=audio_callback, latency="low")
    st.start()
    return st

try:
    stream = start_stream()
except Exception as e:
    print("mic error:", e)
    sys.exit(1)

fig, (ax_time, ax_fft, ax_spec) = plt.subplots(3, 1, figsize=(10, 8)) # создание 3 фигур для грамм

line_time, = ax_time.plot(t_axis, disp_time, lw=1.5)
ax_time.set_xlim(0, OSC_SECONDS)
ax_time.set_ylim(-1, 1)
ax_time.set_title("Oscillogram")
ax_time.set_xlabel("Time, s")
ax_time.set_ylabel("Amplitude")

line_fft, = ax_fft.plot(freqs, np.zeros_like(freqs), lw=1.0)
ax_fft.set_xlim(0, min(MAX_FFT_HZ, SAMPLE_RATE/2))
ax_fft.set_ylim(-100, 0)
ax_fft.set_title("Spectrum (dB)")
ax_fft.set_xlabel("Frequency, Hz")
ax_fft.set_ylabel("dB")

spec_im = ax_spec.imshow( # показывание матрицу spec_data
    spec_data, origin="lower", aspect="auto",
    extent=[0, SPEC_HISTORY, 0, min(MAX_FFT_HZ, SAMPLE_RATE/2)],
    vmin=-100, vmax=0, cmap="magma"
)
ax_spec.set_title("Spectrogram")
ax_spec.set_xlabel("Time → (frames)")
ax_spec.set_ylabel("Frequency, Hz")
fig.colorbar(spec_im, ax=ax_spec, label="dB")

pause_text = ax_time.text(0.5, 0.5, "PAUSED", color="red", fontsize=20,
                          ha="center", va="center", transform=ax_time.transAxes, visible=False)

def screenshot_png():
    name = f"screenshot_{ts()}.png"
    fig.savefig(name, dpi=PNG_DPI)
    print(f"📸 saved {name}")

def on_key(event):
    global paused, paused_t_now
    if event.key == " ":
        paused = not paused
        if paused:
            with state_lock:
                paused_t_now = abs_time_s
        else:
            paused_t_now = None
        pause_text.set_visible(paused)
        fig.canvas.draw_idle()
    elif event.key and event.key.lower() == "p":
        screenshot_png()

fig.canvas.mpl_connect("key_press_event", on_key)

def animate(_):
    if not paused:
        with idx_lock:
            wi = write_idx
            if wi == 0:
                np.copyto(disp_time, ring)
            else:
                tail = OSC_SAMPLES - wi
                np.copyto(disp_time[:tail], ring[wi:])
                np.copyto(disp_time[tail:], ring[:wi])

            if N_FFT <= OSC_SAMPLES:
                np.copyto(fft_time, disp_time[-N_FFT:])
            else:
                fft_time.fill(0.0)
                fft_time[-OSC_SAMPLES:] = disp_time

        line_time.set_ydata(disp_time) #обнровление осцилограммы

        sp = np.fft.rfft(fft_time * window, n=N_FFT)
        db = 20.0 * np.log10(np.abs(sp) + 1e-6)
        line_fft.set_ydata(db)

        global spec_data
        spec_data = np.roll(spec_data, -1, axis=1)
        spec_data[:, -1] = db
        spec_im.set_data(spec_data)

    return line_time, line_fft, spec_im

ani = FuncAnimation(fig, animate, interval=ANIM_INTERVAL_MS, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()

try:
    stream.stop(); stream.close()
except Exception:
    pass