"""Verify -- and if needed repair -- the SNES emulator core.

Why this exists
---------------
The snes9x core shipped in stable-retro's macOS arm64 wheels is built
big-endian on a little-endian CPU. In `cores/snes/port.h`:

    #if defined(__i386__) || ... || defined(__x86_64__) || defined(ARM) || defined(ANDROID)
    #define LSB_FIRST
    #define FAST_LSB_WORD_ACCESS
    #else
    #define MSB_FIRST      <-- Apple Silicon lands here
    #endif

Apple Silicon defines `__aarch64__`/`__arm64__`, none of which appear in that
list, and the Makefile's macOS branch hardcodes `arch = intel` so `ARM` is never
defined either. The core then byte-swaps every word access: the CPU executes but
the game derails and the screen stays black forever.

Rebuilding the same sources with `-DARM` fixes it. `vendor/snes9x_libretro.dylib`
is that rebuild; this module installs it over the broken one.

    python -m smwrl.setup_core          # check, and repair if broken
    python -m smwrl.setup_core --check  # check only, non-zero exit if broken
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDORED = ROOT / "vendor" / "snes9x_libretro.dylib"


def installed_core_path() -> Path:
    import stable_retro

    return Path(stable_retro.__file__).parent / "cores" / "snes9x_libretro.dylib"


def core_is_healthy() -> tuple[bool, str]:
    """Boot the ROM headlessly and check the screen is not black forever.

    A big-endian build renders pure black indefinitely, so a single non-black
    frame within a few seconds of emulated time is a sufficient signal.
    """
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    import stable_retro as retro

    from smwrl.retro_env import GAME, register_integration

    register_integration()
    env = retro.make(
        GAME,
        state=retro.State.NONE,
        inttype=retro.data.Integrations.CUSTOM_ONLY,
        render_mode=None,
    )
    try:
        env.reset()
        noop = np.zeros(12, np.uint8)
        brightest = 0.0
        for _ in range(400):  # ~6.7s of emulated time; the logo appears by ~1.5s
            obs, *_ = env.step(noop)
            brightest = max(brightest, float(obs.mean()))
            if brightest > 1.0:
                return True, f"renders (peak frame brightness {brightest:.1f})"
        return False, f"black screen after 400 frames (peak brightness {brightest:.2f})"
    finally:
        env.close()


def repair() -> bool:
    if not VENDORED.exists():
        print(f"cannot repair: {VENDORED} is missing.\n"
              "Rebuild it with  bash tools/build_snes9x.sh", file=sys.stderr)
        return False
    target = installed_core_path()
    backup = target.with_suffix(".dylib.broken-bigendian.bak")
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(VENDORED, target)
    print(f"installed fixed core -> {target}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only, do not repair")
    args = ap.parse_args()

    ok, why = core_is_healthy()
    print(f"snes9x core: {'OK' if ok else 'BROKEN'} -- {why}")
    if ok:
        return
    if args.check:
        sys.exit(1)

    print("repairing with the little-endian rebuild ...")
    if not repair():
        sys.exit(1)
    ok, why = core_is_healthy()
    print(f"snes9x core after repair: {'OK' if ok else 'STILL BROKEN'} -- {why}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
