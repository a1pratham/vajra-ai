# Phase AI-0 — Colab Environment

## Objective
Set up a reproducible training environment for VAJRA-AI: verify Python,
PyTorch, and CUDA/GPU availability; install dependencies; establish the
project folder structure; and provide a reproducibility (seeding) utility.
No dataset and no model are introduced in this phase.

## Files created / changed
- `configs/config.yaml` — central config (paths, seed, device preference)
- `src/vajra/__init__.py`
- `src/vajra/env.py` — environment report (Python/PyTorch/CUDA/GPU/memory)
- `src/vajra/seed.py` — reproducibility utility
- `src/vajra/config.py` — YAML config loader
- `scripts/smoke_test.py` — CLI smoke test
- `tests/test_env.py` — basic pytest checks
- `notebooks/00_colab_setup.ipynb` — Colab entry point, runs top to bottom
- `requirements.txt`, `.gitignore`, `DATA_SOURCES.md` (template), `data/README.md`

## How to run
See `README.md` sections 2–3. Short version: open
`notebooks/00_colab_setup.ipynb` in Colab and Run All, or run
`python scripts/smoke_test.py` locally after `pip install -r requirements.txt`.

## Expected output
Environment report (Python/PyTorch/CUDA/GPU name/GPU memory) followed by
four `OK` steps and a final `SMOKE TEST: PASSED` line. See README section 4
for the full expected transcript.

## Smoke test
`scripts/smoke_test.py` (also runs inside the notebook). Verifies:
1. Core imports (torch, numpy, yaml)
2. Config loads from `configs/config.yaml`
3. Seeding is reproducible (two seeded runs produce identical random tensors)
4. A minimal tensor forward pass (linear layer) runs on the detected device
   (GPU if available, CPU otherwise — no hard GPU requirement)

## Known limitations
- No dataset access yet — that's Phase AI-1.
- No model, no training loop.
- No checkpoint/resume logic yet — introduced in Phase AI-4, since there is
  nothing to checkpoint in this phase.
- GPU is detected but not required; this phase intentionally works on
  CPU-only Colab or local machines so environment setup is never the
  blocker.

## What to send me if this fails
- The exact cell or command that failed
- The full error/traceback
- Whatever the environment report section printed before the failure
  (even partial output helps)

## Phase completion checklist
- [ ] Notebook runs top to bottom in Colab without manual intervention
- [ ] Environment report prints Python version, PyTorch version, CUDA
      availability, GPU name (or "N/A — running on CPU"), GPU memory
- [ ] `smoke_test.py` prints `SMOKE TEST: PASSED`
- [ ] Reproducibility check passes (seeded runs match)
- [ ] User confirms it works before Phase AI-1 is generated

## Next phase
**Phase AI-1 — Dataset Access** (SEVIR subset, catalog inspection, load and
visualize one event). Not started until this phase is confirmed.
