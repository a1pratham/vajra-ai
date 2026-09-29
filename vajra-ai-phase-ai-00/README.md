# VAJRA-AI — Phase AI-0: Colab Environment

**Project:** VAJRA-AI — AI/ML-based nowcasting of thunderstorm and lightning
(SIH26072)

**Phase objective:** Set up a reproducible training environment. No dataset,
no model yet — this phase only proves that the environment works: Python,
PyTorch, CUDA/GPU detection, dependency installation, project structure, and
a reproducibility (seeding) utility.

Everything in this phase is deliberately small. Later phases build on top of
`src/vajra/` and `configs/config.yaml` without breaking this one.

---

## 1. What's in this package

```
vajra-ai-phase-ai-00/
├── README.md              # this file
├── PHASE.md                # phase objective, checklist, limitations
├── DATA_SOURCES.md         # template — filled in properly in Phase AI-1
├── requirements.txt        # pinned-ish dependencies for Colab + local
├── .gitignore
├── configs/
│   └── config.yaml         # central configuration (paths, seed, device, etc.)
├── notebooks/
│   └── 00_colab_setup.ipynb   # run this top-to-bottom in Colab
├── src/vajra/
│   ├── __init__.py
│   ├── config.py            # loads configs/config.yaml
│   ├── env.py                # environment report (Python/PyTorch/CUDA/GPU)
│   └── seed.py               # reproducibility utility (Python/NumPy/torch)
├── scripts/
│   └── smoke_test.py         # fast CLI smoke test (no notebook needed)
├── tests/
│   └── test_env.py           # basic assertions on env + seeding
├── data/
│   └── README.md             # placeholder — no data in this phase
└── outputs/
    └── .gitkeep               # placeholder — no outputs in this phase
```

## 2. How to run in Google Colab (recommended)

1. Upload this whole folder to Google Drive (e.g.
   `MyDrive/vajra-ai/vajra-ai-phase-ai-00/`), **or** upload just the
   `.zip` and unzip it inside Colab.
2. Open `notebooks/00_colab_setup.ipynb` in Colab.
3. Runtime → Change runtime type → select **GPU** (T4 is fine). GPU is
   optional — the notebook and smoke test both work on CPU too.
4. Run all cells top to bottom (Runtime → Run all).
5. Confirm the final cell prints `SMOKE TEST: PASSED`.

If you uploaded the zip instead of the folder, the first cell in the
notebook includes an `unzip` step — just point it at wherever you uploaded
the zip.

## 3. How to run locally (optional, not required)

```bash
cd vajra-ai-phase-ai-00
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/smoke_test.py
```

## 4. Expected output

Running the smoke test (either in the notebook or via
`python scripts/smoke_test.py`) should print something close to:

```
=== VAJRA-AI Environment Report ===
Python version   : 3.x.x
PyTorch version  : 2.x.x
CUDA available   : True/False
GPU name         : Tesla T4  (or "N/A — running on CPU")
GPU memory (GB)  : 14.7      (or "N/A — running on CPU")
Random seed      : 42
====================================

[1/4] Imports OK
[2/4] Config loaded OK
[3/4] Seed set OK (reproducibility check passed)
[4/4] Tensor forward pass OK on device: cuda (or cpu)

SMOKE TEST: PASSED
```

Exact version numbers will vary by Colab runtime — that's fine. What matters
is that every step prints `OK` and the final line says `PASSED`.

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'yaml'` | `requirements.txt` not installed | Run the pip install cell in the notebook, or `pip install -r requirements.txt` locally |
| `CUDA available: False` on Colab | GPU runtime not selected | Runtime → Change runtime type → GPU, then re-run |
| Notebook can't find `src/vajra` | Wrong working directory | Make sure the notebook's first cell `cd`s into the project root (it does this automatically if you keep the folder structure intact) |
| `smoke_test.py` fails at `[2/4] Config loaded` | `configs/config.yaml` missing or moved | Confirm you unzipped the whole package, not just `notebooks/` |
| Everything fails silently in a fresh Colab session | Runtime was recycled / disconnected | Re-run all cells from the top; this phase has no checkpoints to resume from (that starts in Phase AI-4) |

If something fails that isn't covered here, send me:
- the exact cell/command you ran
- the full error message and traceback
- the output of the environment report section (even if it partially failed)

## 6. What this phase does NOT do

- No dataset access (SEVIR starts in Phase AI-1).
- No model, no training.
- No FastAPI, no frontend.

## 7. Next phase

**Phase AI-1 — Dataset Access.** Will not be generated until you confirm
this phase (AI-0) works in your Colab environment.
