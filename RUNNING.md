# Running an experiment

Training happens on Colab (T4). Analysis and reports happen locally.

---

## 0. Quick reference

### 0.1 To run a specific experiment, set exactly TWO things

Everything else follows from these. There is no third thing to remember.

| | `--config` (the **science**) | `--experiment` (the **folder**) |
|---|---|---|
| **ex1** — one forced-equivalent neuron | `config/default.yaml` | `ex1` |
| **ex2** — each framework out of the box | `config/config_ex2.yaml` | `ex2` |
| scratch / smoke test | either | *omit* → flat in `local_runs/` |

**They must agree with each other.** `--config` decides which neuron gets trained;
`--experiment` decides which folder the numbers land in. Pair them wrongly and you
file one experiment's results under another's name.

`--config` is **required everywhere** and has no default, precisely so this pairing
has to be stated rather than assumed. Every script also **prints the pair it is
using** before doing any work (§0.2), and `collect_results.py` **refuses** to merge
rows that came from a different config (§0.4). Result folders are separated by
construction, so ex1 and ex2 cannot overwrite each other.

On Colab add a third: `--results-root <Drive path>`, or results die with the session.

### 0.2 The full sequence, in order

Set `CFG` and `EXP` from §0.1 once, then run these top to bottom. **Every command is
complete** — nothing is implied.

```powershell
# ─── set these two, then nothing below changes ───────────────────
$CFG = "config/config_ex2.yaml"      # ex1: config/default.yaml
$EXP = "ex2"                         # ex1: ex1
# ─────────────────────────────────────────────────────────────────

# ── 1. what am I running on?  (no args at all)
.venv\Scripts\python check_env.py

# ── 2. is the network identical in all three frameworks?
#      AND are the neuron settings the ones this experiment asked for?
.venv\Scripts\python check_network.py --config $CFG --experiment $EXP --all

# ── 3. do the neurons behave identically?  (writes to $EXP/equivalence/)
.venv\Scripts\python equivalence_check.py --config $CFG --experiment $EXP

# ── 4. is each measurement sane in isolation?  (optional, smoke test)
.venv\Scripts\python check_metrics.py --config $CFG --experiment $EXP

# ── 5. build the dataset cache BEFORE anything is timed
.venv\Scripts\python prepare_data.py --config $CFG --experiment $EXP

# ── 6. the actual runs — 3 frameworks x 3 seeds  (GPU; on Colab, see §3)
.venv\Scripts\python train.py --config $CFG --experiment $EXP `
    --framework snntorch --seed 0

# ── 7. merge Colab output into the experiment folder
.venv\Scripts\python collect_results.py --from <folder> --experiment $EXP

# ── 8. draw all 17 figures
.venv\Scripts\python make_plots.py --experiment $EXP

# ── 9. ex2 only: evidence figure for Norse's alpha bug
.venv\Scripts\python probe_norse_alpha.py --experiment $EXP
```

| # | script | what it answers | needs |
|---|---|---|---|
| 1 | `check_env.py` | what am I running on? | nothing |
| 2 | `check_network.py` | identical network? correct neuron settings? | `--config` |
| 3 | `equivalence_check.py` | do the neurons behave identically? | `--config` |
| 4 | `check_metrics.py` | is each measurement sane? | `--config` |
| 5 | `prepare_data.py` | build the dataset cache first | `--config` + dataset |
| 6 | `train.py` | **the actual run** | `--config` + dataset + **GPU** |
| 7 | `collect_results.py` | merge Colab output | `--from` + `--experiment` |
| 8 | `make_plots.py` | draw all 17 figures | `--experiment` |
| 9 | `probe_norse_alpha.py` | Norse alpha-bug evidence | `--experiment` |

**1–5 are local and need no GPU.** Only 6 needs the GPU; only 5 and 6 need the
dataset.

> **`--experiment` on steps 2, 4 and 5 is a LABEL.** Those scripts write no results,
> so it changes nothing — it just stamps the output with which experiment you were
> checking for, so a scrolled-back terminal still says. Steps 3 and 6–9 use it for
> real, to decide the output folder.

**Every script prints its identity before doing anything** — experiment, config path
and hash, framework where it applies:

```
==========================================================================
check_network.py -- is the network identical across frameworks?
  experiment  ex2   (label only)
  config      config/config_ex2.yaml   hash 20af039c9d78
  framework   all three
  seed        0
==========================================================================
```

So a pasted snippet or a scrolled terminal can never be ambiguous about which
experiment produced it.

### 0.3 Every flag, with its default

**Experiment-specific — these three carry the experiment's identity:**

| flag | default | used by | note |
|---|---|---|---|
| `--config <path>` | **none, REQUIRED** | 2, 3, 4, 5, 6 | the experiment's science |
| `--experiment <exN>` | none → `local_runs/`; required in 7 | 3, 6, 7, 8, 9 | the output folder |
| `--experiment <exN>` | none | 2, 4, 5 | **label only** — those scripts write nothing |
| `--results-root <path>` | `experiments` | 3, 6, 8, 9 | on Colab, point at Drive |

**`train.py`** — the only script with a large surface:

| flag | default | note |
|---|---|---|
| `--framework <name>` | `snntorch` | `snntorch` \| `spikingjelly` \| `norse` |
| `--seed <int>` | config's `training.seed` | overrides the config |
| `--epochs <int>` | config's `training.epochs` | overrides the config; use `1` for a smoke test |
| `--device cuda\|cpu` | config's `training.device` | overrides the config |
| `--max-batches <int>` | all | truncate training — smoke tests only, **never** a real run |
| `--max-eval-batches <int>` | all | same, for evaluation |
| `--no-energy` | off | skip NVML (also skipped automatically if unavailable) |
| `--allow-ephemeral` | off | permit writing outside Drive on Colab. **Overrides a deliberate safety block** |
| `--notes "<text>"` | `""` | free text stored with the run |

**The check scripts:**

| script | flag | default | note |
|---|---|---|---|
| `check_network.py` | `--experiment <exN>` | none | **label only** — stamps the output, changes nothing |
| | `--all` | off | build all three and compare — **normally what you want** |
| | `--framework <name>` | `snntorch` | detail view of one instead |
| | `--seed <int>` | `0` | |
| | `--batch <int>` | `4` | dummy batch; does **not** affect the fingerprint |
| `check_metrics.py` | `--experiment <exN>` | none | **label only** |
| | `--framework <name>` | `snntorch` | |
| | `--device cuda\|cpu` | config | |
| | `--batch <int>` | `8` | |
| | `--latency-samples <int>` | `20` | far below the 100 a real run uses — this is a smoke test |
| | `--idle-seconds <float>` | `3.0` | likewise, vs 20 in a real run |
| `prepare_data.py` | `--experiment <exN>` | none | **label only** — the cache is keyed by dataset settings, not experiment |
| | `--splits train test` | `train test` | which splits to build |
| | `--no-prebuild` | off | fetch one batch instead of writing the whole cache |

**Analysis:**

| script | flag | default | note |
|---|---|---|---|
| `collect_results.py` | `--from <folder>` | **REQUIRED** | the Drive folder or extracted download |
| | `--experiment <exN>` | **REQUIRED** | must match what `--from` holds — now **checked**, see §0.4 |
| | `--dry-run` | off | preview, change nothing |
| | `--force` | off | merge anyway when the guard fires. Only when certain |
| `make_plots.py` | `--results-dir <path>` | derived from `--experiment` | read CSVs from an explicit folder |
| | `--out-dir <path>` | `<experiment>/figures` | disposable preview |
| | `--condition <column>` | `framework` | what goes on the x-axis; lets a later experiment compare variants instead |
| | `--formats png,pdf` | `png` | add `pdf` only if you need vector art |

### 0.4 Can two experiments contaminate each other?

Audited. Short answer: **no, and one gap has been closed.**

| | verdict |
|---|---|
| **result folders** | **Safe by construction.** Every path comes from `output_dirs()`: `<results-root>/<experiment>/{results,equivalence}`. ex1 and ex2 cannot write to the same file. Works identically for a local `experiments/` and a Drive `--results-root`. |
| **folder creation** | **Automatic.** Both `src/results.py` and `equivalence_check.py` call `mkdir(parents=True, exist_ok=True)`, so the folders appear on their own, locally or on Drive. No `mkdir` needed — see §2. |
| **figures** | **Safe.** `make_plots.py` writes to `<experiment>/figures/`, derived from `--experiment` only. |
| **the dataset cache** | **Shared on purpose, and that is correct.** The cache is keyed by the *dataset* settings, and ex1 and ex2 have byte-identical dataset blocks — only the neuron differs. So they share one cache and neither rebuilds it. Change any dataset setting and a *separate* cache is built rather than the old one overwritten. |
| **`collect_results.py`** | **Was the one real gap. Now guarded.** |

**The gap that existed.** `collect_results.py` checked only the CSV *schema*, never
that the incoming rows belonged to the target experiment. `runs.csv` has no
`experiment` column, and row keys are `run_id`s — which carry a timestamp but no
experiment name. So `--from <folder>\ex2 --experiment ex1` would **append** ex2's
rows into ex1's `runs.csv`. They would not collide, they would silently coexist, and
every later mean, figure and conclusion would be computed over a mixture.

**What it does now.** It compares the `config_path` of the incoming rows against the
rows already there. No config in common → different experiments → it refuses:

```
REFUSING TO MERGE: this looks like a different experiment.

  target  experiments/ex1/  was produced by: config/colab.yaml, config/default.yaml
  incoming rows were produced by:            config/config_ex2.yaml

  If --experiment is wrong, fix it. If this really is the same
  experiment run from a renamed config, re-run with --force.
```

Overlap is deliberately allowed, because one experiment can legitimately span several
configs — ex1 has runs from both `default.yaml` and `colab.yaml`, and must stay
mergeable. `--force` overrides after showing the same message as a warning.

Verified four ways: mismatch refuses · a legitimate ex1 re-merge still works and stays
idempotent · `--force` proceeds with a warning · merging into the correct empty ex2
folder works.

`--dry-run` still previews everything without writing.

### 0.5 Expected verdicts — per experiment

A correct result can look like a failure. Check here before debugging.

| | `check_network.py --all` | `equivalence_check.py` |
|---|---|---|
| **ex1** | **PASS** | worst max\|dv\| ≈ **1.2e-07** — float32 epsilon. The neurons were forced to match, and they do |
| **ex2** | **PASS** — neuron changes, weight init does not | worst max\|dv\| ≈ **2.3e-01**, first divergence at timestep 10 — the frameworks' own defaults are different neurons. **This is the ex2 result** |

> **`equivalence_check.py` issues no pass/fail and always exits 0.** It measures the
> deviations, prints them, and draws the plots — the judgement is yours. One
> threshold could not serve both experiments: ex1 wants agreement, ex2 expects
> divergence, and a script printing FAIL for the expected outcome would only train
> you to ignore it. Your `equivalence.tolerance.*` values are still printed beside
> the measured numbers as a reference to eyeball against, and they decide whether the
> detailed first-divergence table is worth showing. Nothing is enforced.
>
> `check_network.py` **does** still issue PASS/FAIL, deliberately: it checks hard
> invariants (identical weights, identical parameter counts) that are either true or
> a bug, and it draws no plots to judge from.

And the weight fingerprint printed by `check_network.py` is **machine-specific**; it
will not match the value in a Colab-run report. See §8.1.

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
```

**No `mkdir` needed.** `src/results.py` and `equivalence_check.py` both create their
output folders with `mkdir(parents=True, exist_ok=True)`, so
`<results-root>/<experiment>/{results,equivalence}` appears on its own.

If you want to confirm Drive is mounted and writable *before* committing to a
15-minute run, check it rather than creating it:

```python
!ls /content/drive/MyDrive || echo "DRIVE NOT MOUNTED"
```

> Avoid `!mkdir -p .../ex1/{results,equivalence}` in a Colab cell. IPython treats
> `{...}` in a `!` line as its own Python interpolation, so bash brace expansion and
> IPython fight over the same syntax. Since the folders are created automatically,
> the whole problem is avoidable.

> The pip warnings about opencv / jax / rasterio are normal: `tonic` requires
> `numpy<2`, Colab ships numpy 2. Nothing we use is affected. Confirm with
> `check_env.py`, not with pip's output.

---

## 3. Run

Three independent choices, none of which live in a config file:

| flag | says | chosen per | required? |
|---|---|---|---|
| `--config` | the experiment's **science** (neuron settings, dataset) | experiment | **YES — always pass it** |
| `--experiment ex2` | **which folder** results go in | run — you name it | no, see below |
| `--results-root <path>` | **where** folders live | machine | no, defaults to `experiments` |

**`--config` is required and has no default.** Every script errors out without it.
That is deliberate. It used to default to `config/default.yaml`, which meant
`train.py --experiment ex2` with no `--config` would quietly train **ex1's neuron**
and file the result under `experiments/ex2/` — a run labelled ex2 that is actually
ex1, with nothing in the output saying so. Now it cannot happen.

So **every experiment takes the same flags**, and only their values change:

| | `--config` | `--experiment` |
|---|---|---|
| ex1 — one forced-equivalent neuron | `config/default.yaml` | `ex1` |
| ex2 — each framework out of the box | `config/config_ex2.yaml` | `ex2` |

Omit `--experiment` and everything lands flat in `local_runs/` — for smoke tests
and scratch work, no naming decision needed. That flag keeps a default because
"no experiment folder" is a meaningful documented choice; "no config" is not.

On Colab always add `--results-root` pointing at Drive. Without it results go to
Colab's temporary disk, and **train.py refuses to start** rather than let you
find out after a 15-minute run.

**Set the experiment once, at the top. Nothing below is hardcoded** — change these
three lines and the whole cell runs a different experiment:

```python
DRIVE = "/content/drive/MyDrive/snn_results"

# ─── the ONLY lines that change per experiment ───
CFG = "config/default.yaml"        # ex1  │  ex2: config/config_ex2.yaml
EXP = "ex1"                        # ex1  │  ex2: ex2
SEEDS = "0 1 2"
# ─────────────────────────────────────────────────

!python check_env.py                                        # versions, GPU, NVML rate
!python check_network.py --config {CFG} --all               # expect OVERALL: PASS
!python equivalence_check.py --config {CFG} --experiment {EXP} --results-root {DRIVE}

!python prepare_data.py --config {CFG}                      # ← ALWAYS. see below

!for s in {SEEDS}; do for fw in snntorch spikingjelly norse; do \
    python train.py --config {CFG} --experiment {EXP} \
      --framework $fw --seed $s --results-root {DRIVE}; \
  done; done
```

`{CFG}` and `{EXP}` are IPython interpolating the Python variables before the shell
sees the line; `$fw` and `$s` are the shell's own loop variables. The two do not
collide, so the loop picks up whichever experiment you set above.

> **Why no braces anywhere else in that cell.** IPython expands `{...}` in a `!`
> line as Python. Anything using bash brace syntax — `{results,equivalence}`,
> `${var}` — gets caught by IPython first. Stick to `$var` in the shell and `{var}`
> for Python, and the two never fight.

**Per-experiment expectations, so a correct result is not mistaken for a fault:**

| | `equivalence_check.py` | `check_network.py --all` |
|---|---|---|
| **ex1** `config/default.yaml` | **PASS** at ~1.2e-07 — the neurons were forced to match | **PASS** |
| **ex2** `config/config_ex2.yaml` | worst max\|dv\| ≈ **2.3e-01** — the frameworks' own defaults are different neurons. **This divergence is the ex2 result** | **PASS** — the neuron changes, weight init does not |

`equivalence_check.py` writes into `<results-root>/<experiment>/equivalence/` using
the same path resolver as `train.py`, so on Colab its figures and summary JSON land
on Drive next to the results and survive a disconnect. **Nothing extra to pass** —
`--experiment` and `--results-root` are all it needs.

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

**At the end of each session**, one cell. `EXP` and `DRIVE` are still set from §3, so
this needs no editing either:

```python
!zip -r {EXP}_from_colab.zip {DRIVE}/{EXP}
from google.colab import files; files.download(f'{EXP}_from_colab.zip')
```

**On the laptop**, extract anywhere and merge:

```powershell
.venv\Scripts\python collect_results.py --from <extracted folder>\ex1 --experiment ex1
```

Same shape for any experiment — `--from <folder>\ex2 --experiment ex2`. The
`--experiment` value must match the folder you are merging into, or you will file
ex2's runs under ex1.

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
| `error: the following arguments are required: --config` | working as intended — pass `--config config/default.yaml` (ex1) or `--config config/config_ex2.yaml` (ex2). There is no default, so a run can never silently use the wrong experiment's neuron |
| `check_network.py` fingerprint doesn't match the one in the report | expected, not a fault. Fingerprints are **machine-specific** — see §8.1. The check is that the three frameworks agree with each other on one machine. |
| `equivalence_check.py` gave no PASS/FAIL | by design — it reports numbers and plots, you decide. Always exits 0, so it is safe to chain. See §8.2. |
| an experiment's `runs.csv` contains rows from a different experiment | `collect_results.py --from` and `--experiment` are independent, and nothing cross-checks them. Merging `<folder>\ex2` with `--experiment ex1` files ex2's runs under ex1. `--dry-run` first if unsure; the `config_path` column is how you spot it afterwards. |

---

## 8. Local checks (no GPU needed)

All four run on the laptop in seconds. None needs a GPU, and only
`prepare_data.py` needs the dataset. Run them after changing a config, the
architecture, or an adapter.

```powershell
.venv\Scripts\python check_env.py
.venv\Scripts\python check_network.py    --config config/default.yaml --all
.venv\Scripts\python equivalence_check.py --config config/default.yaml
.venv\Scripts\python check_metrics.py    --config config/default.yaml
.venv\Scripts\python -m src.results               # results-writer self-test
```

Swap in `config/config_ex2.yaml` to check Experiment 2 instead.

---

### 8.1 `check_network.py` — is the network the same in all three frameworks?

Builds the network and reports what it actually is. **No dataset needed** — it
feeds a dummy tensor of the right shape, so it is instant.

| flag | default | what it does |
|---|---|---|
| `--config <path>` | — | **REQUIRED.** Which experiment's neuron to build. |
| `--all` | off | Build **all three** frameworks and compare them side by side. **This is the one you normally want.** |
| `--framework <name>` | `snntorch` | Build just one and print its layer-by-layer shapes. Used instead of `--all` when you want detail on a single framework. |
| `--seed <int>` | `0` | Which seed's weights to build. |
| `--batch <int>` | `4` | Dummy batch size. Small keeps it fast; it does not affect the fingerprint. |

**The two ways to run it, and when:**

```powershell
# the fairness check — use this after any config or architecture change
.venv\Scripts\python check_network.py --config config/default.yaml --all

# the detail view — use this when adding a framework or debugging shapes
.venv\Scripts\python check_network.py --config config/default.yaml --framework norse
```

**What `--all` verifies.** Five checks, then `OVERALL: PASS` or `FAIL`:

| check | why it matters |
|---|---|
| weight fingerprints identical | all three start from **byte-identical weights**. Without this, no accuracy comparison between frameworks means anything. |
| trainable parameters identical | 18,254. A mismatch means one framework's neuron registers learnable parameters the others don't. |
| flatten size identical | 800, **measured** by a dummy forward rather than hardcoded. |
| output shapes identical | `(batch, 10)` spike counts. |
| reset clears state everywhere | a leftover membrane from the previous sample would silently corrupt every result. |

It also prints the neuron and surrogate per framework under a heading saying
**"these are SUPPOSED to differ"** — so a real difference is never mistaken for a
bug. In ex2 that block is where you see the three out-of-the-box neurons diverge.

**`--all` must PASS for both configs**, including ex2. The neuron changes there, but
weight initialisation does not — so a FAIL on ex2 means something other than the
neuron broke.

> **The fingerprint is machine-specific — this is expected, not a fault.**
> Running this on the laptop gives a *different* fingerprint from the one recorded
> in a Colab run, because PyTorch does not promise identical random numbers across
> torch versions or CPU/CUDA builds. Experiment 1's runs recorded
> `30b4902cd20a39fe` for seed 0 on a Colab T4 (torch 2.11.0+cu128); this laptop
> (torch 2.13.0+cpu) produces `d03b6a70b7398043` for the same seed. **Both are
> correct.** The check is that the three frameworks agree *with each other on one
> machine*, and they do in both cases. Never compare a fingerprint across machines —
> record it per machine.

### 8.2 `equivalence_check.py` — do the neurons behave identically?

One neuron per framework, fed the identical input current, membrane traces compared
timestep by timestep. No network, no dataset.

| flag | default | what it does |
|---|---|---|
| `--config <path>` | — | **REQUIRED.** |
| `--experiment <exN>` | none → `local_runs/` | Where the figures and summary JSON go. |
| `--results-root <path>` | `experiments` | On Colab, point at Drive. |

Verdict depends on which experiment you are checking, and **both verdicts are correct**:

**It issues no verdict and always exits 0.** It prints the measured deviations, shows
your configured reference values beside them, and writes the plots. What a deviation
means depends on the experiment, so the reading is yours:

| config | measured | meaning |
|---|---|---|
| `config/default.yaml` (ex1) | worst max\|dv\| ≈ **1.2e-07** | float32 epsilon — the neurons were forced to match, and they do to the limit of 32-bit arithmetic |
| `config/config_ex2.yaml` (ex2) | worst max\|dv\| ≈ **2.3e-01**, first divergence at timestep 10 | the frameworks' own defaults are genuinely different neurons. **This is the result**, not a problem to fix |

When the deviation exceeds your reference, it also prints a **first-divergence
table** — the timestep where the traces part company and the membrane values on
either side. On ex2 that table shows the input gains directly: 0.150 / 0.075 / 0.015
for the same input, i.e. gains 1.0 / 0.5 / 0.1.

### 8.3 `check_metrics.py` — is each measurement sane in isolation?

Exercises the measurement primitives (timing, throughput, latency, memory, energy,
spike counting) one at a time, so a broken metric is caught before it contaminates a
real run. Takes `--config` (required), plus `--framework`, `--device`, `--batch`,
`--latency-samples`, `--idle-seconds`.

Energy is skipped automatically when NVML or the power sensor is unavailable, which
is the normal case on a laptop.

### 8.4 `check_env.py` — what am I actually running on?

Versions, GPU, CUDA, driver, NVML power-sensor refresh rate, and the Norse build.
Takes no arguments. **Run it first in any new Colab session** — and trust it over
pip's warnings.
