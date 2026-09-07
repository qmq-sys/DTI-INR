# DTI-INR

This project is an independent implementation of DTI-INR.
It does not reuse the previous QINR/INR/PINR implementation.

## Research question

Can an Implicit Neural Representation (INR) directly represent a spatial DTI
parameter field and recover DTI parameters from diffusion signals via the
classical DTI forward model?

$$
S(x,g,b) = S_0(x)\,\exp\!\left(-b\,g^\top D(x)\,g\right)
$$

Two models are planned:

| Model | Name | Idea |
|-------|------|------|
| **A** | Spatial DTI-INR | \(x \rightarrow \{S_0(x), D(x)\}\) |
| **B** | Spatial-Angular DTI-INR | Keep \(D=D(x)\); treat \((g,b)\) as continuous q-space queries |

A classical **WLS-DTI** fit is the mandatory baseline.

## Project status

| Phase / Task | Status |
|--------------|--------|
| Phase 0 — New project | **PASS (Task 1)** |
| Phase 1 — DTI forward model | **PASS (Task 1)** |
| Phase 2 — WLS-DTI baseline | **PASS (Task 1)** |
| Task 2 — Spatial DTI-INR (A) network | Not started |
| Task 3+ — training / HCP / B | Not started |

## Environment

```bash
cd DTI-INR
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Required packages: Python, PyTorch, NumPy, SciPy, NiBabel, DIPY, Matplotlib, PyYAML.

## Task 1 — run unit tests

From the project root:

```bash
python -m tests.test_dti_forward
python -m tests.test_wls_dti
```

## Directory layout

```text
DTI-INR/
├── configs/
├── data/             # placeholders (Task 4)
├── models/           # placeholders (Task 2 / Task 6)
├── physics/          # DTI forward model  (Task 1)
├── baselines/        # WLS-DTI           (Task 1)
├── training/         # placeholders (Task 3+)
├── evaluation/       # placeholders
├── visualization/    # placeholders
├── experiments/
├── scripts/
└── tests/
```

## Principles

- Start from a clean project; do not import prior QINR / INR / PINR code.
- First version: Fourier features + MLP + DTI physics + MSE only.
- Do not skip the WLS baseline.
- Correctness > interpretability > reproducibility > completeness > performance.
- Do not implement Model A/B until Task 1 passes and later tasks are requested.
