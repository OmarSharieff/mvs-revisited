# Multi-View Stereo Revisited
Runnable AI-assisted partial reproduction of Goesele, Curless and Seitz, CVPR 2006.

## Status before submission
The implementation, tests and experiments are provided. This is NOT yet an
instructor-approved exact reproduction. Get agreement on projective fusion and
synthetic evaluation. Add all three names/contact details and review the report.
Do not claim AI-generated code as individual student implementation work. Each
member should understand, test and improve it and record their actual contributions.

## Setup and reproduce
Tested on Linux with Python 3.11. Use WSL on Windows.
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run.sh
```
The default run tests the implementation, renders 48x48 synthetic views, runs six
experiments, downloads TempleSparseRing, reconstructs 16 real views at 160x120,
and regenerates report.pdf. Internet is needed for initial dataset acquisition.
All project paths are relative to the root; no source edits are needed.
For an offline synthetic-only run:
```
python code/main.py --suite --size 48
python code/report.py
```
For tests alone: `python -m pytest tests -q`.

## Results and interpretation
Each results/<run>/ folder contains mesh.ply, mesh.png (when nonempty), maps.npz,
diagnostics.png and metrics.json. The saved JSON files are the measured expected
outputs; compare geometry metrics within roughly 1e-5 on the same environment.
Runtimes depend on hardware and include stereo only. Real-data geometric error
is unavailable, not zero. Coverage is valid foreground pixel coverage, not mesh
completeness. summary.json and metrics.csv describe the most recent invocation;
per-run metrics.json files retain all experiment records.

## Modules
- code/mvs.py: camera geometry, sampling, NCC, votes, search, weighted fusion.
- code/synthetic.py: calibrated textured cube renderer and ground-truth depths.
- code/main.py: experiments, Middlebury camera parsing, results and visualization.
- code/report.py: evidence-based report regeneration.
- tests/test_core.py: independent numeric and geometric checks.

## Important limitations
Fusion uses a 4mm projective distance grid instead of VripPack. The synthetic
coarse step is 5mm; Temple uses 2.5mm. Images are intentionally reduced resolution.
No original benchmark accuracy or full paper reproduction is claimed. The synthetic
cube tests one controlled geometry; it is not an independently sampled real benchmark.
Core runtime is CPU based. Read the accompanying project guide for explanations.

## Team review
Assign geometry/search to one member, patch/NCC/voting to another, and fusion/
evaluation to the third. Each must review another module and explain the whole
pipeline. Add genuine contributions to contributions.csv. Complete names, TA
scope agreement and oral rehearsal before submitting. The due date in the supplied
instructions is 21 December 2026. Send algorithm.txt separately as instructed.

## Measured expected results

| Run | Depth MAE mm | Foreground coverage | Stereo seconds |
|---|---:|---:|---:|
| baseline | 5.352 | 0.7524 | 16.0 |
| real | N/A | N/A | 140.9 |
| textureless | 18.842 | 0.2602 | 17.3 |
| threshold04 | 8.092 | 0.9104 | 20.3 |
| threshold08 | 2.069 | 0.48 | 18.1 |
| window3 | 4.847 | 0.8914 | 8.1 |
| window9 | 9.183 | 0.584 | 48.4 |

Hardware and measured environment: results/environment.json. This run used CPU only. Allow several minutes for the full suite on similar hardware; downloading and PDF rendering add time.
# Multi-Stereo-Revisited-
