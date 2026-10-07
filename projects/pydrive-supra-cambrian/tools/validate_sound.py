"""Headless validation of the engine-sound DSP (supra/sound.py).

Runs the EngineSynth offline across rpm/throttle/gear sweeps, checks the output
is finite, DC-free, and spectrally sane (the 787B's dominant tonal energy must
track the four-rotor firing frequency rpm*4/60), benchmarks the synth against
the real-time budget, and renders listenable WAV sweeps.

Usage: python3 tools/validate_sound.py [outdir]
"""
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supra.sound import EngineSynth, SR, BLOCK, MAZDA_787B_GEAR_RATIOS  # noqa: E402

FAILED = []


def gate(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


class Spec:
    def __init__(self, name, redline, ratios):
        self.name = name
        self.redline_rpm = redline
        self.gear_ratios = ratios


class Veh:
    def __init__(self, spec):
        self.spec = spec
        self.speed = 0.0
        self.rpm = 900.0
        self.boost = 0.0
        self.gear = 1
        self.slip_angle = 0.0
        self.wheel_sr = [0.0, 0.0, 0.0, 0.0]


CARS = {
    "mazda787b": Spec("mazda787b", 9000.0, list(MAZDA_787B_GEAR_RATIOS)),
    "supra": Spec("supra", 6800.0, [3.2, 2.1, 1.5, 1.1, 0.9, 0.75]),
    "rx7": Spec("rx7", 8000.0, [3.4, 2.0, 1.4, 1.0, 0.8]),
    "lr4": Spec("lr4", 6000.0, [3.6, 2.2, 1.5, 1.0, 0.8, 0.7]),
    "porsche_919evo": Spec("porsche_919evo", 8500.0,
                           [3.09221, 2.31930, 1.77249, 1.36942, 1.10771, 0.92796, 0.82357]),
}


def render(car, seconds, drive_fn):
    """drive_fn(t) -> (rpm, throttle, gear, speed, screech). Returns full buffer."""
    spec = CARS[car]
    veh = Veh(spec)
    synth = EngineSynth()
    out = []
    nblocks = int(seconds * SR / BLOCK)
    for b in range(nblocks):
        t = b * BLOCK / SR
        rpm, thr, gear, speed, screech = drive_fn(t)
        veh.rpm = rpm
        veh.gear = gear
        veh.speed = speed
        veh.wheel_sr = [0.0, 0.0, screech, screech]
        synth.update(veh, thr)
        out.append(synth.synthesize(BLOCK))
    return np.concatenate(out)


def save_wav(path, buf):
    buf = np.clip(buf, -0.99, 0.99)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((buf * 32767).astype(np.int16).tobytes())


def dominant_freq(buf, lo=80.0):
    win = buf * np.hanning(len(buf))
    spec = np.abs(np.fft.rfft(win))
    freqs = np.fft.rfftfreq(len(buf), 1.0 / SR)
    spec[freqs < lo] = 0.0
    return float(freqs[int(np.argmax(spec))])


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    print("== basic sanity: all cars, full sweep ==")
    for car in CARS:
        redline = CARS[car].redline_rpm

        def drive(t, redline=redline):
            f = min(t / 6.0, 1.0)
            rpm = 1200 + f * (redline - 1200)
            gear = 1 + int(f * 4.99)
            return rpm, 1.0, gear, f * 90.0, 0.2 if 0.4 < f < 0.6 else 0.0

        buf = render(car, 7.0, drive)
        gate(f"{car}: finite", bool(np.all(np.isfinite(buf))))
        gate(f"{car}: bounded", bool(np.max(np.abs(buf)) <= 1.0), f"peak={np.max(np.abs(buf)):.3f}")
        gate(f"{car}: not silent", bool(np.sqrt(np.mean(buf ** 2)) > 0.01), f"rms={np.sqrt(np.mean(buf**2)):.3f}")
        # per-block DC (blocker runs per buffer)
        blocks = buf[: len(buf) // BLOCK * BLOCK].reshape(-1, BLOCK)
        gate(f"{car}: DC-free per block", bool(np.max(np.abs(blocks.mean(axis=1))) < 1e-3))
        save_wav(outdir / f"engine_{car}_sweep.wav", buf)

    print("== 787B firing-frequency tracking ==")
    for rpm in (3000.0, 5000.0, 7000.0, 9000.0):
        veh = Veh(CARS["mazda787b"])
        veh.rpm = rpm
        veh.speed = 50.0
        veh.gear = 4
        synth = EngineSynth()
        chunks = []
        for _ in range(40):  # let ramps settle, then measure
            synth.update(veh, 1.0)
            chunks.append(synth.synthesize(BLOCK))
        steady = np.concatenate(chunks[20:])
        fire = rpm / 60.0 * 4.0
        peak = dominant_freq(steady)
        # dominant peak must sit on a harmonic of the firing frequency
        harm = peak / fire
        on_harm = abs(harm - round(harm)) < 0.06 and 1 <= round(harm) <= 20
        gate(f"787B {int(rpm)}rpm: peak {peak:.0f}Hz is a firing harmonic (fire={fire:.0f}Hz, h={harm:.2f})", on_harm)

    print("== overrun stays loud (open pipes) ==")
    veh = Veh(CARS["mazda787b"])
    veh.rpm = 7500.0
    veh.speed = 70.0
    veh.gear = 4
    s = EngineSynth()
    on = [s.synthesize(BLOCK) for _ in range(20) if s.update(veh, 1.0) or True]
    rms_on = float(np.sqrt(np.mean(np.concatenate(on[10:]) ** 2)))
    off = [s.synthesize(BLOCK) for _ in range(20) if s.update(veh, 0.0) or True]
    rms_off = float(np.sqrt(np.mean(np.concatenate(off[10:]) ** 2)))
    gate("787B overrun >= 45% of full-throttle level", rms_off >= 0.45 * rms_on,
         f"on={rms_on:.3f} off={rms_off:.3f}")

    print("== wind/road audible at speed, absent at rest ==")
    def quiet_drive(speed):
        veh = Veh(CARS["mazda787b"])
        veh.rpm = 2200.0
        veh.speed = speed
        veh.gear = 6
        s = EngineSynth()
        buf = []
        for _ in range(20):
            s.update(veh, 0.0)
            buf.append(s.synthesize(BLOCK))
        return float(np.sqrt(np.mean(np.concatenate(buf[10:]) ** 2)))
    rms_still = quiet_drive(0.0)
    rms_fast = quiet_drive(90.0)
    gate("env noise louder at 90 m/s than at rest", rms_fast > rms_still * 1.15,
         f"still={rms_still:.4f} fast={rms_fast:.4f}")

    print("== realtime budget ==")
    veh = Veh(CARS["mazda787b"])
    veh.rpm = 8000.0
    veh.speed = 80.0
    veh.gear = 5
    s = EngineSynth()
    s.update(veh, 1.0)
    s.synthesize(BLOCK)  # warm
    t0 = time.perf_counter()
    reps = 200
    for _ in range(reps):
        s.update(veh, 1.0)
        s.synthesize(BLOCK)
    per_block_ms = (time.perf_counter() - t0) / reps * 1000.0
    budget_ms = BLOCK / SR * 1000.0
    gate(f"synth per block {per_block_ms:.2f}ms < 30% of {budget_ms:.1f}ms budget",
         per_block_ms < 0.30 * budget_ms)

    # Realistic driving lap snippet WAV (accel through gears, brake, downshift)
    def lap(t):
        cyc = t % 10.0
        if cyc < 6.0:                       # accelerate 1->5
            f = cyc / 6.0
            gear = 1 + int(f * 4.99)
            rpm = 3500 + ((f * 5) % 1.0) * 5200
            return rpm, 1.0, gear, 20 + f * 70, 0.0
        else:                               # brake + downshifts
            f = (cyc - 6.0) / 4.0
            gear = max(1, 5 - int(f * 4.0))
            rpm = 8600 - ((f * 4) % 1.0) * 3200
            return rpm, 0.0, gear, 90 - f * 60, min(0.5, f)
    save_wav(outdir / "engine_mazda787b_lap.wav", render("mazda787b", 20.0, lap))
    print(f"\nWAVs written to {outdir.resolve()}")

    if FAILED:
        print(f"\n{len(FAILED)} gate(s) FAILED: {FAILED}")
        sys.exit(1)
    print("\nall gates green")


if __name__ == "__main__":
    main()
