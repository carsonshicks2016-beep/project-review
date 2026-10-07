#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OUT=${1:-"$ROOT/.rally/audio-review/offline-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$OUT"
mcs -out:"$OUT/audiocheck.exe" "$ROOT/Tools/audio-check/Shim.cs" \
    "$ROOT/Assets/Audio/SFX/ProceduralAudio.cs" "$ROOT/Assets/Audio/SFX/AudioGate.cs" \
    "$ROOT/Assets/Audio/SFX/AudioEventQueue.cs" "$ROOT/Assets/Audio/SFX/AudioMixProfile.cs" \
    "$ROOT/Assets/Audio/SFX/EngineSynth.cs" "$ROOT/Assets/Audio/SFX/RallyEngineVoice.cs" \
    "$ROOT/Assets/Audio/SFX/TyreSynth.cs" "$ROOT/Assets/Audio/SFX/WindSynth.cs" \
    "$ROOT/Assets/Audio/SFX/ImpactSynth.cs" "$ROOT/Assets/Audio/SFX/ForestAmbience.cs" \
    "$ROOT/Tools/audio-check/Program.cs"
cd "$OUT"
status=0
mono audiocheck.exe > checks.txt || status=$?
cat checks.txt
exit "$status"
