"""Laboratory-grade console dashboard for BLADE curriculum training.

Append-only (robust across terminals and through `tee`): a header panel, a
periodic instrument row with an inline stage-promotion gauge, colour-coded stage
banners on promotion, and a final summary.  ANSI width is avoided by never
padding *coloured* strings — pad first, colour second.
"""
from __future__ import annotations

import sys
import time

_TTY = sys.stdout.isatty()


def _c(code):
    return (lambda s: f"\033[{code}m{s}\033[0m") if _TTY else (lambda s: str(s))


DIM, BOLD = _c("2"), _c("1")
CYAN, GREEN, YEL, RED, MAG, BLU, WHT = (_c("96"), _c("92"), _c("93"), _c("91"),
                                        _c("95"), _c("94"), _c("97"))


def bar(frac, width=10):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "█" * n + "░" * (width - n)


def _hms(sec):
    sec = int(max(sec, 0))
    if sec < 90:
        return f"{sec}s"
    if sec < 5400:
        return f"{sec // 60}m"
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}"


class Dashboard:
    def __init__(self, run, loadout, device, total_steps, stages):
        self.t0 = time.time()
        self.total_steps = total_steps
        self.stage_names = [s.name for s in stages]
        rule = "━" * 76
        print()
        print(CYAN("┏" + rule + "┓"))
        print(CYAN("┃ ") + BOLD(WHT("B L A D E")) + DIM("   curriculum trainer") +
              " " * 42 + DIM("lab") + CYAN(" ┃"))
        print(CYAN("┗" + rule + "┛"))
        print("  " + DIM("run ") + WHT(f"{run:<14}") + DIM("loadout ") + MAG(f"{loadout:<14}") +
              DIM("device ") + WHT(f"{device:<6}") + DIM("budget ") + WHT(f"{total_steps/1e6:.0f}M steps"))
        print("  " + DIM("curriculum  ") + DIM(" → ").join(MAG(n) for n in self.stage_names))
        print()
        print(DIM("   iter  │ stage          │ promotion        │ throughput          │ instruments"))

    def row(self, it, stage_idx, stage_name, steps, sps, metrics, promote_frac):
        eta = _hms((self.total_steps - steps) / max(sps, 1))
        stg = MAG(f"S{stage_idx+1}/{len(self.stage_names)}") + " " + BOLD(WHT(f"{stage_name:<7}"))
        gauge = GREEN(bar(promote_frac)) + YEL(f" {int(promote_frac*100):3d}%")
        thru = f"{steps/1e6:5.1f}M {sps/1e3:4.1f}k/s ETA {eta:>4}"
        m = metrics
        inst = ("surv " + YEL(f"{m.get('survive',0):.2f}") +
                "  reach " + YEL(f"{m.get('reach',0):.2f}") +
                "  hit " + YEL(f"{m.get('hits',0):.1f}") +
                "  ret/s " + GREEN(f"{m.get('ret',0):+.2f}") +
                "  ent " + BLU(f"{m.get('ent',0):.1f}") +
                "  kl " + DIM(f"{m.get('kl',0):.3f}"))
        print(f"  {it:5d}  {DIM('│')} {stg} {DIM('│')} {gauge} {DIM('│')} {thru}  {DIM('│')} {inst}")

    def promote(self, frm, to, it, metric_name, val):
        print(GREEN("  " + "═" * 72))
        print("  " + BOLD(GREEN(f"✓ PROMOTED  {frm} → {to}   ·   iter {it}   ·   {metric_name} {val:.2f}")))
        print(GREEN("  " + "═" * 72))

    def banner(self, text, color=CYAN):
        print(color("  " + "─" * 72))
        print("  " + BOLD(color(text)))
        print(color("  " + "─" * 72))

    def done(self, out, final_stage):
        print(CYAN("  " + "═" * 72))
        print("  " + BOLD(WHT(f"✦ training complete  ·  {_hms(time.time()-self.t0)}  ·  reached stage {final_stage}")))
        print("  " + DIM(f"checkpoint → {out}"))
        print(CYAN("  " + "═" * 72))
