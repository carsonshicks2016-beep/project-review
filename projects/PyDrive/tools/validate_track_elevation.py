"""Track elevation validation — Gate 1 of PHYSICS_3D_PLAN.md.

Checks the Stage 1 height-field work without touching physics:

  * defaults + levers — ordinary/procedural tracks are flat by default;
                        Nordschleife keeps real elevation unless --flat is
                        requested; --hills / --hill-scale still opt into and
                        scale procedural profiles; layouts never change.
  * caps & floors     — with hills=True, every style honors its grade cap and
                        crest launch-speed floor (HILLS table) across
                        difficulties and seeds. Because grade/vcurv are
                        loop-periodic central differences, these bounds cover
                        the wrap indices too — a seam in the profile would
                        blow the cap instantly, so this doubles as the
                        periodicity check.
  * hill_scale        — 0 kills the profile, 1.5 stays within 1.5x amplitude.
  * ridge             — the canonical jump track has EXACTLY two jumpable
                        crest clusters, launch speeds in the designed band,
                        well separated along the arc.
  * determinism       — two fresh subprocesses generate identical hill
                        profiles for every named/special track (the
                        _stable_seed discipline extended to elevation).
  * bank_corners      — unit test of the (unwired) banking helper: sign,
                        saturation, straights.
  * ctor contract     — explicit elevation arrays are accepted; mismatched
                        lengths raise.

Usage: PYTHONPATH="$PWD" python3 tools/validate_track_elevation.py
Exits non-zero on any failure.
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supra.track import (G, HILLS, NAMED, Track, _stable_seed, bank_corners,
                         configure_hills, make_track, named_track, stadium)

FAILURES = []


def check(name: str, ok: bool, detail: str = ""):
    tag = "ok  " if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def lerp(pair, t):
    return pair[0] + (pair[1] - pair[0]) * t


# --------------------------------------------------------------------------- #
def test_defaults_and_flat_lever():
    print("flat defaults + Nordschleife real elevation + --flat/--hill-scale:")
    configure_hills(enabled=False, scale=1.0, force_flat=False)
    for name in ("club", "akina", "tech", "oval2"):
        t = named_track(name)
        check(f"{name}: flat by default", abs(t.elev_gain) < 1e-6,
              f"elev_gain={t.elev_gain:.1f}m")
    nord = named_track("nordschleife")
    check("nordschleife keeps real elevation by default",
          nord.elev_gain > 250.0, f"elev_gain={nord.elev_gain:.1f}m")
    t = named_track("akina")
    x, y = t.center[10]
    fr = t.frame(float(x), float(y))
    check("frame() carries live z/grade keys",
          fr["z"] == float(t.z[10]) and fr["grade"] == float(t.grade[10]))
    check("stadium flat by default", abs(stadium().elev_gain) < 1e-6)
    try:
        configure_hills(enabled=False, force_flat=True)        # the --flat lever
        for name in ("club", "akina", "ridge", "nordschleife"):
            tf = named_track(name)
            check(f"--flat: {name} completely flat",
                  not np.any(tf.z) and not np.any(tf.grade)
                  and np.all(np.isinf(tf.launch_speed)))
        configure_hills(enabled=True, scale=0.5, force_flat=False)   # the --hill-scale lever
        th = named_track("akina")
        configure_hills(enabled=True, scale=1.0, force_flat=False)
        full = named_track("akina")
        check("--hill-scale 0.5 shrinks the profile",
              0.0 < np.ptp(th.z) < 0.8 * np.ptp(full.z),
              f"ptp {np.ptp(th.z):.1f}m vs {np.ptp(full.z):.1f}m")
        check("same seed, same layout regardless of hills",
              bool(np.array_equal(th.center, full.center)))
    finally:
        configure_hills(enabled=True, scale=1.0, force_flat=False)


def test_caps_and_floors():
    print("grade caps + launch floors (hills=True), incl. wrap indices:")
    for style, p in HILLS.items():
        for diff in (0.0, 0.5, 1.0):
            for seed in (1, 42, 777):
                t = make_track(diff, style, seed=seed, hills=True)
                mg = float(np.max(np.abs(t.grade)))
                vmin = float(np.min(t.launch_speed))
                floor = lerp(p["launch_floor"], diff)
                amp = lerp(p["amp"], diff)
                ok = (mg <= p["grade_cap"] * 1.001
                      and vmin >= floor * 0.999
                      and np.ptp(t.z) <= amp * 1.05)
                check(f"{style} d={diff} seed={seed}", ok,
                      f"max|grade|={mg:.4f}/{p['grade_cap']}  "
                      f"min_launch={vmin:.1f}/{floor:.1f}  "
                      f"ptp={np.ptp(t.z):.1f}/{amp:.1f}m")
    # hilly touge actually HAS hills
    t = make_track(1.0, "touge", seed=3, hills=True)
    check("touge d=1.0 has real elevation", t.elev_gain > 4.0,
          f"elev_gain={t.elev_gain:.1f}m")


def test_hill_scale():
    print("hill_scale lever:")
    t0 = make_track(0.8, "touge", seed=5, hills=True, hill_scale=0.0)
    check("hill_scale=0 -> flat", not np.any(t0.z))
    t1 = make_track(0.8, "touge", seed=5, hills=True, hill_scale=1.5)
    amp = lerp(HILLS["touge"]["amp"], 0.8) * 1.5
    check("hill_scale=1.5 within 1.5x amplitude",
          np.ptp(t1.z) <= amp * 1.05, f"ptp={np.ptp(t1.z):.1f}/{amp:.1f}m")
    a = make_track(0.8, "touge", seed=5, hills=False)
    b = make_track(0.8, "touge", seed=5, hills=True)
    check("hills on/off -> identical 2D layout",
          bool(np.array_equal(a.center, b.center)))


def test_ridge():
    print("ridge — the canonical jump track:")
    t = named_track("ridge")
    check("has elevation by design", t.elev_gain > 3.0,
          f"elev_gain={t.elev_gain:.1f}m")
    check("touge grade cap honored",
          float(np.max(np.abs(t.grade))) <= HILLS["touge"]["grade_cap"] * 1.001)

    jump = t.launch_speed < 36.0
    idx = np.where(jump)[0]
    if len(idx) == 0:
        check("jumpable crests exist", False, "no launch_speed < 36 m/s")
        return
    # group contiguous indices into clusters (loop-aware)
    breaks = np.where(np.diff(idx) > 1)[0]
    clusters = np.split(idx, breaks + 1)
    if len(clusters) > 1 and idx[0] == 0 and idx[-1] == len(t.center) - 1:
        clusters[0] = np.concatenate([clusters[-1], clusters[0]])
        clusters.pop()
    check("exactly TWO jumpable crest clusters", len(clusters) == 2,
          f"found {len(clusters)}")
    speeds = [float(t.launch_speed[c].min()) for c in clusters]
    check("launch speeds in the designed band [24, 36] m/s",
          all(24.0 <= v <= 36.0 for v in speeds),
          f"speeds={[f'{v:.1f}' for v in speeds]}")
    if len(clusters) == 2:
        s0 = float(t.arc[clusters[0][len(clusters[0]) // 2]])
        s1 = float(t.arc[clusters[1][len(clusters[1]) // 2]])
        d = abs(s0 - s1)
        d = min(d, t.length - d)
        check("crests well separated", d >= 100.0, f"{d:.0f}m apart")


def test_determinism():
    print("cross-process determinism (the _stable_seed discipline):")
    snippet = (
        "import sys, numpy as np; sys.path.insert(0, '.')\n"
        "from supra.track import NAMED, make_track, named_track, _stable_seed\n"
        "for n, (style, diff, length) in sorted(NAMED.items()):\n"
        "    t = make_track(diff, style, length, seed=_stable_seed(n), hills=True)\n"
        "    print(n, float(np.sum(t.z)).hex())\n"
        "for n in ('ridge', 'speedbowl', 'circuit'):\n"
        "    print(n, float(np.sum(named_track(n).z)).hex())\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = [subprocess.run([sys.executable, "-c", snippet], cwd=root,
                           capture_output=True, text=True) for _ in range(2)]
    ok = all(o.returncode == 0 for o in outs) and outs[0].stdout == outs[1].stdout
    detail = "" if ok else (outs[0].stderr or outs[1].stderr or "stdout mismatch")[:200]
    check("two fresh processes agree on every profile", ok, detail)


def test_bank_corners():
    print("bank_corners helper (unwired; data-path unit test):")
    t = named_track("club")
    b = bank_corners(t, max_deg=8.0)
    check("magnitude saturates at max_deg",
          float(np.max(np.abs(b))) <= np.radians(8.0) * 1.001)
    i = int(np.argmax(t.curvature))   # strongest LEFT turn
    check("left turn banks right-edge-up (negative)", b[i] < 0.0,
          f"bank@maxcurv={np.degrees(b[i]):.2f}deg")
    # the straights check needs a track with REAL straights (club is all
    # sweepers) — and the probe must sit MID-straight: |curvature| is exactly
    # 0.0 all along a dead straight, so argmin would return the first tie,
    # right at the arc junction. Heavy smoothing finds the deepest point.
    ts = named_track("speedbowl")
    bs = bank_corners(ts, max_deg=8.0)
    ksm = np.abs(ts.curvature)
    for _ in range(60):
        ksm = 0.5 * ksm + 0.25 * (np.roll(ksm, 1) + np.roll(ksm, -1))
    j = int(np.argmin(ksm))
    check("straights stay near flat",
          abs(bs[j]) < 0.1 * float(np.max(np.abs(bs))),
          f"bank@mid-straight={np.degrees(bs[j]):.2f}deg")
    check("finite everywhere", bool(np.all(np.isfinite(b))))


def test_ctor_contract():
    print("Track ctor contract:")
    t = named_track("club")
    M = len(t.center)
    z = np.sin(np.linspace(0, 2 * np.pi, M, endpoint=False)) * 5.0
    t2 = Track(t.center.copy(), width=t.width, elevation=z)
    check("explicit elevation accepted + derived fields built",
          np.array_equal(t2.z, z) and np.any(t2.grade)
          and np.isfinite(t2.launch_speed).any())
    try:
        Track(t.center.copy(), width=t.width, elevation=z[:-3])
        check("mismatched elevation length raises", False, "no exception")
    except ValueError:
        check("mismatched elevation length raises", True)


if __name__ == "__main__":
    for fn in (test_defaults_and_flat_lever, test_caps_and_floors,
               test_hill_scale, test_ridge, test_determinism,
               test_bank_corners, test_ctor_contract):
        fn()
    if FAILURES:
        print(f"\nFAIL: {len(FAILURES)} check(s) failed: {FAILURES}")
        sys.exit(1)
    print("\nPASS: track elevation validated (Gate 1)")
