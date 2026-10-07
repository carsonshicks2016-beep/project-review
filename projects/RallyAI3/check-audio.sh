#!/usr/bin/env bash
# Offline render and check of the procedural audio, without opening Unity.
#
# WHY THIS EXISTS
#
# The synths under Assets/Audio/SFX generate every sample themselves, and the failure
# modes are all silent ones: a voice that never sounds, a resonator pinned against the
# soft clipper so everything is a square wave, a level curve that runs away across the
# rev range, a source that hums when the car is stationary. None of that shows up in a
# compile, and in the editor it needs ears and a rebuilt scene to notice.
#
# So the REAL synth sources are compiled against a small Unity shim (Tools/audio-check/
# Shim.cs) and driven through their real lifecycle. The DSP is never copied — the files
# in Assets are the files under test, so this cannot drift away from what ships. The
# shim only stands in for the engine API and the physics components' field names, which
# means renaming any field the synths read breaks this build immediately.
#
# Usage:  ./check-audio.sh
# Exit:   0 = every check passed
# Also writes listenable WAVs to Tools/audio-check/out/ — stage_run.wav is the mixed one.
set -uo pipefail
cd "$(dirname "$0")"

command -v csc  >/dev/null || { echo "MISSING: csc (install Mono)";  exit 1; }
command -v mono >/dev/null || { echo "MISSING: mono";                exit 1; }

OUT="Tools/audio-check/audiocheck.exe"

csc -nologo -langversion:latest -out:"$OUT" \
    Tools/audio-check/Shim.cs Tools/audio-check/Program.cs \
    Assets/Audio/SFX/*.cs || { echo "HARNESS BUILD FAILED"; exit 1; }

cd Tools/audio-check && mono audiocheck.exe
