"""Stage 1.10 acceptance: both seeds render to a non-blank image.

Rendering needs an OpenGL context, which a headless CI box may lack; if no
context is available this test SKIPS (counts as pass) rather than failing. Where
GL works, it renders each seed and asserts the PNG is written and not blank.

Runs:  python3 tests/test_render.py   (requires mujoco + a GL context)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import mujoco

from personal_cambrian.seeds import SEEDS
import view_creature

_GL_OK = None


def _gl_available() -> bool:
    global _GL_OK
    if _GL_OK is None:
        try:
            m = mujoco.MjModel.from_xml_string("<mujoco/>")
            mujoco.Renderer(m, 64, 64)
            _GL_OK = True
        except Exception:  # noqa: BLE001 - no GL context in this environment
            _GL_OK = False
    return _GL_OK


def test_seeds_render_to_nonblank_png():
    if not _gl_available():
        print("  (skipped: no GL context)")
        return
    with tempfile.TemporaryDirectory() as d:
        for name, fn in SEEDS.items():
            out = os.path.join(d, f"{name}.png")
            img = view_creature.render(fn(), out)
            assert os.path.exists(out) and os.path.getsize(out) > 1000
            with open(out, "rb") as f:
                assert f.read(8) == b"\x89PNG\r\n\x1a\n"     # valid PNG signature
            assert img.var() > 5 and int(img.max()) > 20     # something was drawn


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
