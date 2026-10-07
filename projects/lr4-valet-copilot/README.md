# LR4 Valet Copilot

Prototype interface for a car-specific Raspberry Pi assistant for a 2015 Land Rover LR4 HSE.

Current vehicle assumption: US-market 2015 LR4 HSE with the 3.0L supercharged V6. Confirm against the VIN/build sheet before turning mock thresholds into maintenance guidance.

Open `index.html` in a browser to run the mock interface. The app is silent by default, wakes on a configurable phrase, speaks with browser text-to-speech when available, and simulates Bluetooth OBD plus IMU/accelerometer streams.

## Intended Architecture

- Raspberry Pi runs the local collector, dashboard, voice gate, text-to-speech, and local storage.
- Bluetooth OBD-II adapter streams generic PIDs first: RPM, coolant temp, intake air temp, fuel trims, MAF, throttle, speed, voltage if supported.
- IMU/accelerometer module connects over I2C and measures launch acceleration, brake decel, cornering load, pitch, roll, and road harshness.
- Local SQLite database stores sessions, baselines, delta events, service notes, and user preferences.
- The dashboard is hosted locally by the Pi so a phone, tablet, or in-car screen can connect without cloud service.
- The LLM summarizes sensor packets only. Deterministic thresholds still decide warning state so the model is never the safety-critical layer.

## Local LLM Layer

The UI now has an **LLM Core** panel with four paths:

- Mock Pi LLM: deterministic fallback for testing before the Pi exists.
- Pi valet proxy: recommended in-car mode. The browser calls `/api/llm/brief` on the Pi, and the Pi talks to the local model.
- Ollama local API: direct browser-to-Ollama mode for development on the same machine.
- llama.cpp server: direct browser-to-OpenAI-compatible llama.cpp mode for development.

Why the proxy matters: if a phone opens the dashboard, `127.0.0.1` is the phone, not the Raspberry Pi. Serving the dashboard and `/api/llm/brief` from the Pi avoids that trap.

Run the prototype Pi proxy:

```bash
python3 pi_valet_server.py --host 0.0.0.0 --port 8787 --model llama3.2:1b
```

Then open:

```text
http://lr4-valet.local:8787
```

The proxy expects Ollama at `http://127.0.0.1:11434` by default. For a Pi 5, start with a small 1B-class model for short spoken briefs; a 3B-class quantized model may be acceptable if latency is tolerable. On a Zero 2 W, keep the rule-based fallback or use it as a collector that talks to a stronger Pi.

## LR4-Specific Focus

- Cooling behavior after idle and low-speed crawling.
- Battery voltage reserve for accessory-heavy use.
- Fuel trim drift as an early signal for intake or sensor issues.
- Air suspension feel inferred from pitch, roll, and road harshness deltas.
- Transfer case / low-range usage can be added later with enhanced Land Rover-specific data if the adapter and PID source support it.

## Hardware Direction To Validate Before Buying

- Raspberry Pi 5 if you want a fast dashboard and local voice stack.
- Raspberry Pi Zero 2 W if you want a smaller, cheaper, lower-power collector and are comfortable with lighter voice features.
- OBDLink MX+ or similar quality Bluetooth OBD-II adapter with sleep behavior.
- 6-DoF IMU such as an LSM6DSOX breakout for acceleration and gyro.
- Small USB or I2S microphone, compact speaker, fused 12V-to-5V power supply, and a tidy under-dash enclosure.

Reference checks from current product pages:

- Raspberry Pi 5 lists Bluetooth 5.0 / BLE, dual-band Wi-Fi, a 40-pin header, and 5V/5A USB-C power.
- Raspberry Pi Zero 2 W lists Bluetooth 4.2 / BLE, 2.4GHz Wi-Fi, and a tiny 65mm x 30mm form factor.
- OBDLink MX+ lists automatic sleep/wake behavior, Bluetooth, 8-18V operating range, and low-power mode.
- Adafruit LSM6DSOX lists 3-axis acceleration plus 3-axis gyro with I2C/SPI support.

## Build Phases

1. Keep this mock UI and tune the exact prompts, voice cadence, and LR4 watchlist.
2. Run `pi_valet_server.py` and test the LLM Core through the Pi valet proxy.
3. Connect Bluetooth OBD and record a baseline drive.
4. Add IMU calibration and chassis-aligned mounting.
5. Replace mock thresholds with Carson's actual LR4 baselines.
