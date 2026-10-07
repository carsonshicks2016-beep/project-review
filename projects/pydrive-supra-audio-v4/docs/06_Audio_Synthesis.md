# Audio Synthesis

## Audio V4 contract and Doppler invariant

Audio V4/V5-contract is the default procedural engine. Set `SUPRA_AUDIO_ENGINE=v3` for the
one-release fallback. V4 receives versioned, sequence-locked `AudioFrame`
telemetry without altering physics, Fable observations, rewards, or checkpoints.

Doppler is required. World sources use phase-continuous fractional propagation
delay at 343 m/s within 350 m and must never receive a second pitch multiplier.
Listener-local wind is excluded. Camera teleports use a 100 ms transition so a
cut cannot become a synthetic pitch sweep. Runtime remains entirely procedural;
recordings and imported game samples are not calibration inputs, runtime assets,
or release gates. Mechanical invariants and deterministic listening scenarios
define acceptance.

The V5 contract carries engine lifecycle, four wheel contacts/surfaces, brake
temperature, suspension travel/velocity, impact corner, and shift phase. The
renderer gives the exhaust, intake, engine, transaxle, chassis, brakes, and four
wheel contacts independent positions and directivity. Dense fields use a
deterministic nearest-voice budget to retain callback headroom.

Run `tools/render_audio_v4_scenarios.py` to produce the procedural listening
suite and blank scoring matrix under `runtime/`. Run `tools/validate_audio_v4.py`
for mechanical, Doppler, directivity, reset, allocation-growth, and 44.1/48/96
kHz runtime gates.

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
