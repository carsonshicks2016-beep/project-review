# Reproduction and sharing boundaries

The repository contains source, tests, experiment specifications and selected evidence. It is not an independently validated one-command installation. Existing macOS launchers target Carson's local `/Users/REVIEW_USER/FlyGarden` environment. Several scripts also depend on local artifacts omitted from Git. Static source review is available immediately; running the complete experiment requires restoring its pinned inputs.

## Upstream inputs

See `reports/provenance.json` for exact revisions and SHA-256 checksums.

- Neural model: https://github.com/eonsystemspbc/fly-brain at `a3db62f9436074e485c0278290c2164ed6150808`, locally under `vendor/fly-brain`.
- Body: https://github.com/NeLy-EPFL/flygym at `38c8ec61034cd59bc5ba0de20688d4a3c0000d60`, locally under `vendor/flygym`.
- Annotations: https://github.com/flyconnectome/flywire_annotations at `a83b2776d60d5764cef36b927f5f9679c16c47a2`. Publication attribution is retained in `data/annotations-README.md`. Standalone redistribution terms were not confirmed, so the annotation table is omitted.

Restore the graph artifacts and annotation table at their original relative paths and verify the recorded checksums. The installed environment used Python 3.12; `requirements.lock` records dependencies, including a local FlyGym installation reference that must be adjusted to the restored checkout. The test suite contains data-dependent tests and operational checks; not every test can run on this lightweight snapshot.

## Frozen protocols

Experiment protocols hash source files and input artifacts. Running a historical protocol against changed files is deliberately rejected. Do not remove those checks merely to get a run to start. Historical receipts attest to a particular local evidence package; omitted raw artifacts prevent independent verification of the entire package from Git alone.

The running adaptation sweep is a committed snapshot. Its progress continues locally after publication; GitHub does not update automatically. A completed result should be published in a later commit with its audit and provenance.

## Licensing and attribution

Application code is GPL-2.0-or-later, as recorded in the existing LICENSE and README. Dependencies retain their own licenses; Three.js's notice is retained in `static/vendor`. The separate published-equation diagnostic has GPL-3.0-or-later terms described in the existing learning-rule audit. Upstream datasets, model code and morphology are not claimed as original work. Omitted vendor repositories should be obtained from their upstream sources with their notices intact.
