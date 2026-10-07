# Legacy Observatory (pre-remake)

Frozen 2026-07-23 when the from-scratch Observatory remake started. This is
the exact working Three.js client that previously lived at repo-root
`observatory/`.

- The remake is now active at **`/observatory/`** via `observatory/dist`.
- This tree is **rollback/reference only** — do not edit for new features.
- Python playback (`supra/observatory.py`) and transport
  (`command-center/observatory_api.py`) remain shared.

To rebuild this archive for comparison:

```bash
cd legacy/observatory
npm install
npm run build
```

Note: asset builder (`tools/build_observatory_assets.py`) writes to the remake
tree (`observatory/public/...`). Copy assets manually if you need a fresh
legacy public tree.
