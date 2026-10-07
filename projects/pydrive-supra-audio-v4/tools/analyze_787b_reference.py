#!/usr/bin/env python3
"""Extract non-reconstructive calibration features from a user-owned PCM WAV."""
from __future__ import annotations
import argparse, json, wave
from pathlib import Path
import numpy as np

def analyze(path: Path):
    with wave.open(str(path), "rb") as w:
        sr=w.getframerate(); channels=w.getnchannels(); width=w.getsampwidth()
        if width != 2: raise ValueError("only 16-bit PCM WAV is supported")
        x=np.frombuffer(w.readframes(w.getnframes()),np.int16).astype(float)/32768
    x=x.reshape(-1,channels).mean(1); block=max(256,int(sr*.05)); usable=len(x)//block*block
    frames=x[:usable].reshape(-1,block); rms=np.sqrt(np.mean(frames*frames,axis=1)+1e-12)
    crest=np.max(np.abs(frames),axis=1)/(rms+1e-12); window=np.hanning(block)
    spectrum=np.abs(np.fft.rfft(frames*window,axis=1)); freqs=np.fft.rfftfreq(block,1/sr); bands=[]
    for lo,hi in ((40,200),(200,500),(500,1000),(1000,2000),(2000,4000),(4000,8000),(8000,min(16000,sr/2))):
        mask=(freqs>=lo)&(freqs<hi); bands.append(float(np.mean(spectrum[:,mask])) if mask.any() else 0)
    total=max(sum(bands),1e-12)
    return {"schema":"mazda787b-derived-profile-v1","sample_rate":sr,"duration_s":len(x)/sr,
      "loudness_rms_percentiles":np.percentile(rms,[10,50,90]).round(6).tolist(),
      "crest_factor_db_percentiles":(20*np.log10(np.percentile(crest,[10,50,90]))).round(3).tolist(),
      "normalized_spectral_bands":[round(v/total,7) for v in bands],"contains_reconstructive_audio":False}

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("wav",type=Path); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
    a.out.write_text(json.dumps(analyze(a.wav),indent=2)+"\n")
