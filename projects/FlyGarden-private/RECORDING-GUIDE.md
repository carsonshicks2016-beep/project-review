# Watching a recorded fly and brain

Open Fly Garden using its existing launcher. Choose **Recordings** in the right-hand panel, then **Inspect fly + individual neurons** on a recorded run.

New embodied runs retain all modeled neuron spike events and target 30 physical pose samples per simulated second. The physics sampler uses the actual 0.5 ms gait boundaries, without changing gait or neural timing. Playback interpolates rigid body poses between saved samples; neural activity uses the saved spike timestamps. World events, sensory inputs, and motor decisions retain their original 100 ms update schedule.

Use Play, the time slider, and playback speed to explore a run. Drag on the brain to orbit it. Select a neuron from the list or click its branches: its root ID, annotation, run spike count, current rate, recent spike raster, and full-run firing-rate history appear below. Search by root ID or cell type, filter by annotated population, and load further batches of 128 shapes. “Show active neurons only” changes the display, not the recorded network.

The heat map uses one fixed **0–500 Hz** scale across runs, with a preceding 100 ms window. Spike flashes fade over 100 simulated milliseconds. The first 100 ms of a recording have a partial activity window; earlier activity is not invented. These are display encodings. This point-neuron model assigns one activity value to all of a neuron’s anatomical branches; it does not simulate electrical propagation along them.

All imported neurons’ spike events are saved. The initial interactive view loads 128 of the most active available skeletons; default videos request 256. Counts distinguish the full modeled network from the visible anatomy. Shapes are real FlyWire v783 branching skeletons fetched by exact root ID and cached locally. Selecting a neuron restores its full branching detail. Missing shapes remain explicitly unavailable. Entire-network anatomy is not downloaded upfront.

Every finalized new trial queues a 1080p, 30 FPS H.264 MP4, including capture/fall/interrupted trials with complete samples. Neural-only conditioning recordings say that no physical body was simulated. Baseline and older recordings do not invent brain activity. During an experiment batch, videos wait for simulation and archival to finish. One worker renders the queue. Opening playback pauses the interactive arena, while an independent experiment continues.

The video list shows queued, rendering, complete, failed, or interrupted. Retry a failed export from the playback page. “Render MP4 with this selection” creates a new export with the chosen arena camera and population, reusing raw data. It uses up to 256 neurons in that population; it does not reproduce arbitrary manual per-neuron selections or your interactive orbit angle.

A video is not a checkpoint. Keep the raw recordings to inspect or rerender them. Use **Save complete scene** and **Branch** to resume or branch modeled state. Saved scenes from before this upgrade still load. Scrubbing does not restore a checkpoint or alter the fly.

## Storage and recovery

New media defaults to `/Users/REVIEW_USER/FlyGarden/data`. To put new recordings, cached anatomy, and exports on another drive, set `FLYGARDEN_MEDIA_DIR` to an absolute folder before launching. Existing default recordings and exports remain discoverable; files are never automatically migrated or deleted. Existing queued jobs keep their original export folder.

The writer checks space before each simulation window and disk write, preserving a 2 GiB free-space reserve. If storage fills, it pauses and reports the reason. Free space before resuming. Raw events are published in compressed chunks with hashes; metadata commits complete samples. After interruption, only committed samples are played, while unfinished tails remain untouched. At startup, recordings with an identified dead writer are marked interrupted and queued for video automatically. Export retries reread raw data. Completed exports keep snapshots of their renderer sources; the duplicate temporary render input is removed after successful encoding.

Offline export requires the installed Google Chrome at its standard macOS path. `FLYGARDEN_CHROME` can specify another compatible Chromium executable. The pinned Playwright dependency drives the application’s offline renderer; FFmpeg is supplied by the existing imageio-ffmpeg dependency.

## Interpretation

A video can reveal activity, movement, avoidance, contacts, and conditioning events. It does not establish biological fidelity, behavioral learning, consciousness, or preservation of the original animal’s identity. The neural controller, plasticity mechanism, and supplied gait were kept fixed for this recording phase.

Spike flashes fade over 15 simulated milliseconds. Inactive branches are nearly dark; selecting a neuron preserves its activity color. The heat map still averages over 100 milliseconds. Rapidly firing neurons can remain bright even with short flashes. Existing MP4s retain their original appearance; rerender to apply the new display.

Recorded arena playback now offers Free camera: drag to orbit, right-drag to pan, scroll to zoom. Save viewpoint stores a numbered view for that recording in this browser; choose it from Saved viewpoints to restore it, including after reopening the page. Scrubbing and playback preserve the free camera. MP4 exports from Free camera use its current position and target as a fixed viewpoint. The brain retains its independent orbit controls. No neural recomputation is required.

The recording timeline includes clickable event markers, an event menu, and Previous/Next event controls. Choosing an event pauses playback at its saved simulation timestamp and preserves a free camera viewpoint. Odor markers use display thresholds of 0.15 on and 0.10 off; obstacle markers indicate a sensory ray below 2 mm, not physical collision. Threat markers use the existing engineered threat signal above 0.25. Food/contact markers refer to the game model's food-contact event; they do not claim a measured biological feeding response. Reinforcement and capture markers appear only when recorded. Recording-end markers distinguish a scheduled diagnostic boundary from a game outcome.

To repeat the longer recording diagnostic, open Learn → Reproducible experiments → Record full-brain obstacle trial · 30 s. It uses seed 801, the level-2 arena (food, one interior wall plus boundaries), unchanged full-brain decoding and supplied gait, and frozen plasticity. It stops on a physical/game termination or at 30 simulated seconds. The scheduled boundary is not a successful foraging outcome. Each run is saved separately and queues its own video.

Follow playback now chooses a clear viewing side when an arena block would hide the fly. This affects only the observer camera; sensory occlusion, physics, neural activity and saved events stay as recorded. Free-camera and saved-view modes keep the viewpoint you choose.
