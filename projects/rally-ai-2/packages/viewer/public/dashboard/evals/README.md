# Bundled eval reports

Copy Phase D `*.eval.json` files from `packages/sim/runs/` into this directory,
then add their filenames to `index.json`:

```json
{
  "files": ["checkpoint_000100.pt.eval.json"]
}
```

Vite copies this directory unchanged. The dashboard also accepts the original
files directly through its file picker or drop target, so copying is optional.
