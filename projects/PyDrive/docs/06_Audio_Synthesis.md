# Audio Synthesis

## Audio V5 procedural contract

Audio V5 is the default; set `SUPRA_AUDIO_ENGINE=v3` for the one-release
fallback. Its sequence-locked telemetry contract is isolated from physics,
Fable observations, rewards, and checkpoints. Runtime audio is entirely
procedural and real recordings or imported game samples are not release gates.

World sources use phase-continuous propagation-delay Doppler at 343 m/s within
350 m and must never receive a second pitch transform. Exhaust, intake, engine,
transaxle, chassis, brakes, and four wheel contacts have independent positions
and directivity. Camera teleports use a 100 ms equal-power transition and
listener-local wind bypasses propagation.

Run `tools/render_audio_v4_scenarios.py` for deterministic listening renders and
`tools/validate_audio_v4.py` for mechanical, spatial, reset, allocation-growth,
and 44.1/48/96 kHz runtime gates.

## Observatory immersive stream

The Fable Five Observatory builds on the same telemetry-driven vehicle stems,
selecting the Mazda 787B or Porsche 919 Evo voice from validated checkpoint
identity. Exhaust, intake, machinery, transaxle, MGU/turbo, four tyre contacts,
brakes and chassis events are positioned independently around the vehicle.
Trackside, drone, free and broadcast cameras receive distance attenuation,
source directivity, propagation-delay Doppler, camera-cut continuity and a low
early-reflection bed. Roof/onboard cameras switch to the close mechanical mix.

An additional deterministic Nordschleife atmosphere supplies forest wind,
foliage, day birds, dusk/night insects and camera-motion wind. Dry, dusk and
night visual presets are forwarded to audio, but these presentation controls
remain downstream of simulation. Track proximity supplies surface state;
wheel slip/load, braking energy, suspension-load change, landing force,
RaceBox clutch/shift phase, turbo boost and 919 MGU/SOC come from authoritative
telemetry.

The server sends framed 48 kHz stereo signed-16 PCM in 1,024-frame chunks. The
AudioWorklet primes a 6,144-frame (128 ms) jitter buffer and fades underrun
boundaries to suppress clicks while remaining below the A/V skew target. The
frame header carries matching simulation sequence and time. Browser camera
updates affect only the audio listener and ambience; they cannot alter physics,
sensors, observations, rewards or policy inference. Autoplay denial, malformed
audio or an underrun produces a visible muted state while visuals continue.

Supra Drift does not use pre-recorded audio samples for the vehicles. Instead, it features a **real-time additive synthesis engine** (`supra/sound.py`) driven directly by the simulation's telemetry via the `sounddevice` library. 

## Engine & Exhaust Sounds

The audio engine produces dynamic sounds that accurately reflect the state of the vehicle:

- **Harmonic Additive Synthesis**: The core engine tone is generated using inline-six (or quad-rotor for the 787B) harmonics that harden and shift dynamically under load.
- **Overrun & Backfires**: When lifting off the throttle, the exhaust note changes (open pipes stay loud, the spectrum tilts to an exhaust thump). Downshifting produces synchronized exhaust crackles and bangs.
- **Turbo Dynamics**: Forced induction models feature a high-pitched turbo whine that spools with boost, and a distinct turbo flutter ("stu-tu-tu-tu" BOV sound) upon throttle lift.
- **Driveline Thump & Whine**: Dogbox shifts produce a transmission chirp and driveline thump, while coasting generates gear whine.
- **V8 Asymmetry**: Piston cars like the LR4 or F150 feature asymmetric exhaust clipping and a V8 idle lope.

## Environment & Interaction Sounds

The audio extends beyond just the engine to encompass the vehicle's interaction with the environment:

- **Tyre Squeal**: Scaled dynamically by the slip angle and slip ratio of the tyres.
- **Kerb Strikes**: Riding over track kerbs produces a distinct thwack/rattle sound synchronized with a camera impulse kick in Viewer V2.
- **Wind & Road Noise**: Speed-squared scaled gusting wind brightens with speed, and road rumble roughens with tyre slide.

## Technical Implementation

- **Mixer Subprocess**: The audio generation runs in a robust, battle-hardened mixer subprocess utilizing a shared memory array layout to pull telemetry from the main simulation loop without blocking it.
- **Graceful Degradation**: The simulation degrades gracefully. If no audio device is found, it simply runs in silence without crashing.
- **Mute**: The audio can be disabled via the `--no-audio` CLI flag or muted live in the viewer by pressing **M**.

Automated tools (`tools/validate_sound.py`) enforce that the DSP blocks remain bounded, DC-free, and execute within strict latency budgets (e.g., ~0.8 ms per block against a 46 ms budget).
