# Running an experiment

Training happens on Colab (T4). Analysis and reports happen locally.

---

## 1. Local setup (once)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python check_env.py
```

CPU-only on purpose. The laptop is for writing code and running
`equivalence_check.py`; the GPU work is on Colab.

---

## 2. Colab session

```python
!git clone -b setup https://github.com/haseebravian756/Benchmark_SNN_Frameworks.git
%cd Benchmark_SNN_Frameworks
!pip install -q -r requirements.txt        # numpy conflict warnings are expected

from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/snn_results/ex1/{results,equivalence}
```

> The pip warnings about opencv / jax / rasterio are normal: `tonic` requires
> `numpy<2`, Colab ships numpy 2. Nothing we use is affected. Confirm with
> `check_env.py`, not with pip's output.

---

## 3. Run

Three independent choices, none of which live in a config file:

| flag | says | chosen per |
|---|---|---|
| `--config` | the experiment's **science** (neuron settings, dataset) | experiment |
| `--experiment ex2` | **which folder** results go in | run — you name it |
| `--results-root <path>` | **where** folders live | machine |

Omit `--experiment` and everything lands flat in `local_runs/` — for smoke tests
and scratch work, no naming decision needed.

On Colab always add `--results-root` pointing at Drive. Without it results go to
Colab's temporary disk, and **train.py refuses to start** rather than let you
find out after a 15-minute run.

```python
DRIVE = "/content/drive/MyDrive/snn_results"

!python check_env.py                                     # versions, GPU, NVML rate
!python check_network.py --all                           # expect OVERALL: PASS
!python equivalence_check.py --experiment ex1 --results-root {DRIVE}

!python prepare_data.py                                  # ← ALWAYS. see below

!for s in 0 1 2; do for fw in snntorch spikingjelly norse; do \
    python train.py --experiment ex1 --framework $fw --seed $s \
      --results-root /content/drive/MyDrive/snn_results; \
  done; done
```

A different experiment changes only `--config` and `--experiment`:

```python
!python train.py --config config/norse_super.yaml --experiment ex6 \
    --framework norse --results-root {DRIVE}
```

### Results accumulate on their own

`runs.csv` is **append-only**. Within one Drive folder everything piles up
automatically — several frameworks in a loop, separate cells, or a reconnect and
a second session on the same account. You never merge those by hand.

Rows are written at the *end* of a run, so a killed run leaves no row rather
than a broken one.

### ⚠️ `prepare_data.py` is not optional

Tonic builds its cache lazily. Skip this and the **first** framework to run
absorbs the whole 60,000-sample conversion — it looks catastrophically slow for
a reason that has nothing to do with the framework.

This actually happened: snnTorch seed 1 recorded 953 s for epoch 1 versus ~128 s
for epochs 2–5, and a training time of 1468 s instead of ~643 s.

Colab wipes `/content` between sessions, so **re-run it after every reconnect**:

```python
!du -sh cache/ 2>/dev/null || echo "CACHE MISSING - run prepare_data.py"
```

---

## 4. Bring results home

**Your laptop is the single source of truth.** Each Colab account has its own
Drive, so hopping between free accounts produces several *partial* result sets.
The laptop is where they come together.

```
account A, session 1  →  ex1: snntorch, norse      ─┐
account B, session 2  →  ex1: spikingjelly         ─┼→  experiments/ex1/
account A, session 3  →  ex1: seed 1               ─┘
```

**At the end of each session**, one cell:

```python
!zip -r ex1_from_colab.zip /content/drive/MyDrive/snn_results/ex1
from google.colab import files; files.download('ex1_from_colab.zip')
```

**On the laptop**, extract anywhere and merge:

```powershell
.venv\Scripts\python collect_results.py --from <extracted folder>\ex1 --experiment ex1
```

It prints every run the folder now holds, so you see at a glance what is there
and what is missing.

> **What the merge does:** it adds only rows your local folder doesn't already
> have. It cannot delete, overwrite or reorder a row. Re-merging the same zip
> changes nothing. That is what makes account-hopping safe — a plain file copy
> would wipe out the runs from the other account.

`--dry-run` previews without writing.

---

## 5. Draw the plots

Once the results are home, one command draws every figure:

```powershell
.venv\Scripts\python make_plots.py --experiment ex1
```

Reads `experiments/ex1/results/{runs,epochs,layers}.csv`, writes 17 PNGs to
`experiments/ex1/figures/`. No GPU, a couple of seconds.

**Any other experiment — change one word:**

```powershell
.venv\Scripts\python make_plots.py --experiment ex2
```

Safe to re-run at any time. Figures are overwritten, never appended, so after a new
seed arrives you just run it again and everything updates together.

| flag | default | what it does |
|---|---|---|
| `--experiment exN` | — | which folder to read and write. **Usually the only flag you need.** |
| `--results-root <path>` | `experiments` | where experiment folders live; change only on another machine |
| `--results-dir <path>` | — | read CSVs from an explicit folder instead (scratch data, manual download) |
| `--out-dir <path>` | `<experiment>/figures` | write figures elsewhere — good for a disposable preview |
| `--condition <column>` | `framework` | which column goes on the x-axis; lets a later experiment compare variants or neuron types instead of frameworks |
| `--formats png,pdf` | `png` | PNG is what the report embeds; add `pdf` only if you need vector art |

```powershell
# preview without touching the experiment folder
.venv\Scripts\python make_plots.py --experiment ex1 --out-dir C:\Temp\figs

# plot scratch runs that never got an experiment folder
.venv\Scripts\python make_plots.py --results-dir local_runs --out-dir local_runs\figures
```

**Two messages that are normal, not errors:**

- `skipped <name> — required columns absent` — that metric is not in this
  experiment's CSVs. The other figures still get drawn.
- `note: only one seed present` — the paired figures (F2) and every ±SD interval
  need two or more seeds.

If one figure fails it prints `FAILED` with the reason and the rest still get drawn.

**Figures are never edited by hand.** Everything is regenerated from the CSVs, so
any figure can be traced back to the rows that produced it. The design, the six
figure families and the published sources behind each convention are in
`local_docs/plotting_schema.md` — read §0 there for the full flag reference.

---

## 6. Layout

```
experiments/ex1/          <- created by --experiment ex1
  report.md               the write-up
  equivalence/            equivalence figures + pass/fail JSON
  figures/                <- created by make_plots.py --experiment ex1
    F0.2_*.png            fairness evidence
    F1.*.png              per-metric distributions
    F2.*.png              paired within-seed views
    F3.*.png              profile and trade-offs
    F4.*.png              training dynamics
    F5.*.png              per-layer structure
    F6.*.png              measurement quality
  results/
    runs.csv              one row per (framework, seed)
    epochs.csv            one row per epoch
    layers.csv            one row per LIF layer
    runs/*.json           full config snapshot per run

local_runs/               <- no --experiment: flat, timestamped, gitignored
```

Output path = `<results-root>/<experiment>/{results,equivalence}`.
**No config file mentions folders.** You name the experiment when you run it.

---

## 7. Things that will bite you

| symptom | cause |
|---|---|
| one framework's training time is 2–3× the others | cache was cold — run `prepare_data.py` |
| results vanished after a Colab reconnect | forgot `--results-root` — train.py now blocks this |
| `runs.csv` has fewer rows than expected | Colab session restarted and began a fresh CSV; `collect_results.py` merges them |
| energy warnings printed after a run | the measurement is not trustworthy; the message says why |
| `idle_power_cold_w` > `idle_power_after_train_w` | back-to-back runs — only the session's first run gets a genuinely cold baseline |

---

## 8. Local checks (no GPU needed)

```powershell
.venv\Scripts\python check_network.py --all       # frameworks start from identical weights
.venv\Scripts\python equivalence_check.py         # neurons behave identically
.venv\Scripts\python check_metrics.py             # each measurement in isolation
.venv\Scripts\python -m src.results               # results-writer self-test
```
