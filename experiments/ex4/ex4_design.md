# Experiment 4 — Design: SpikingJelly in its fastest mode

**Status: built and verified, runs in progress. Written 2026-08-12.**

> **The one-sentence version.** ex1 and ex2 measured SpikingJelly with its fused CUDA
> kernel switched off, so its speed figures there are a floor rather than a fair
> number; ex4 turns the kernel on, confirms the answers do not change, and measures
> what the speed actually is.

This is open item **E5**, and it closes limitation **L2** which is declared in both
the ex1 and the ex2 reports.

Claims marked **[installed]** were verified by reading or running the installed
`spikingjelly 0.0.0.0.14`, not from a web page. Claims marked **[measured]** come
from a run on this project's own hardware, with the command that produced them named.

**Companion document.** The engineering — how the code is isolated, what breaks
silently, the full execution flow — is in
`local_docs/spikingjelly_multistep_intro.md`. This document is the *experiment*: what
question it asks, what the arms are, and what you can and cannot claim from the
results. Read that one if you are touching the code; read this one if you are writing
the report.

---

## 1. Where this experiment sits

### 1.1 The three experiments, in one line each

| | Question | What varies |
|---|---|---|
| **ex1** | Do three frameworks compute the same neuron identically? | the framework, neuron forced identical |
| **ex2** | What do you get if you just use each framework's defaults? | the framework AND its default neuron |
| **ex4** | How fast is SpikingJelly when you let it use its own fast path? | **only the execution strategy** |

ex1 and ex2 both compare *frameworks*. ex4 does not — it compares **one framework
against itself**. That difference matters for what you may claim, and §5 is about
nothing else.

### 1.2 Words used in this document

**Timestep loop.** Our network turns one event recording into T = 20 frames and feeds
them to the layers. Something has to iterate over those 20 frames.

**Single-step (`step_mode='s'`).** *Our* Python `for` loop does the iterating, handing
each layer one frame at a time. This is what `src/network.py` has always done, and
what produced every ex1 and ex2 number.

**Multi-step (`step_mode='m'`).** The whole `[20, batch, C, H, W]` tensor is handed to
the layer at once, and the layer iterates internally.

**Backend.** *How* the layer does that internal iteration:
- `backend='torch'` — a Python loop, same as ours.
- `backend='cupy'` — one compiled CUDA kernel covering all 20 timesteps.

**Fused kernel.** The `cupy` one. "Fused" because 20 separate GPU operations become
one, so the GPU is launched once instead of twenty times.

**BPTT.** Backpropagation through time — the gradient has to travel back through all
20 timesteps. SpikingJelly fuses this direction too, and that is half the speedup.

### 1.3 Why this is not just an optimisation ticket

Two reasons it is a real experiment rather than a chore.

**It fixes an unfairness in ex1 and ex2.** Those reports say SpikingJelly was faster
than snnTorch and Norse. That comparison held SpikingJelly back: it was running in
the only mode our network supported, not the mode its authors intend. Reporting a
framework's speed while disabling its headline feature is a criticism a reader could
fairly make of ex1 and ex2, and ex4 is the answer to it.

**It separates two things that are usually conflated.** The literature quotes fused
CUDA kernels as a framework advantage. But "give the layer all timesteps at once" and
"compile those timesteps into one kernel" are different changes, and only one of them
is fast. ex4 measures them separately (§3), which is something the framework's own
documentation does not do.

---

## 2. The question, stated precisely

> Holding the neuron, the network, the data, the optimiser, the seed and the machine
> fixed, **how much faster does SpikingJelly train when it uses its fused CUDA
> kernel, and does the result change?**

Two halves, and the second one is not a formality:

- **Speed** is the measurement.
- **Equivalence** is the acceptance test. The fused kernel is supposed to compute the
  same neuron. If accuracy moves, the port is wrong and the speed number is
  meaningless. §6 makes this concrete.

### 2.1 What ex4 does NOT ask

- It does not ask whether SpikingJelly is faster than snnTorch or Norse **in this
  mode**. It cannot: they have no comparable kernel here, so that is not a controlled
  comparison. §5.1.
- It does not ask whether the fused kernel is more accurate. It is bit-identical, so
  the question is void — and that is a result, not an absence of one.
- It does not ask about inference latency. The kernel is training-only. §5.3.

---

## 3. The arms — and why there are two

Running only the fast config would produce a number you could not attribute. It
changes **two** things against ex2 at once:

1. the network stops looping over timesteps (restructured forward pass), and
2. the neuron switches to a fused CUDA kernel.

A single run cannot tell you which produced the speedup. So there are two arms.

| Arm | Config | `step_mode` | `backend` | What it isolates |
|---|---|---|---|---|
| **baseline** | `config/config_ex2.yaml` | `s` | `torch` | already run — ex2's SpikingJelly rows |
| **control** | `config/config_ex4_control.yaml` | `m` | `torch` | restructuring **only**, no kernel |
| **fast** | `config/config_ex4_multistep.yaml` | `m` | `cupy` | restructuring **plus** the kernel |

Then:

```
control - baseline  =  cost or benefit of restructuring alone
fast    - control   =  the fused kernel's own contribution
```

**[measured]** `probe_spikingjelly_multistep.py`, Colab T4, one LIF layer at the real
training shape `[T=20, N=128, 12, 30, 30]`:

| Arm | ms/iteration | vs baseline |
|---|---|---|
| baseline — looped `'s'` + torch | 52.32 | 1.00× |
| control — `'m'` + torch | 52.55 | **1.00×** |
| fast — `'m'` + cupy | 10.48 | **4.99×** |

So the expectation for the full runs is that **the control arm is not faster than
ex2**. A null result there is the point of the arm: it is what licenses the sentence
"the entire speedup is attributable to CUDA fusion, not to restructuring the forward
pass." Without it, that sentence is an assumption.

### 3.1 A useful side effect of the control arm

The control arm needs **no cupy and no library patch** (§7.1 — `np.int` is only read
on the cupy path), and it runs on CPU. So it is the clean, unpatched, portable arm,
and it is how the multi-step network was tested locally without a GPU. Worth one
sentence in the report: the restructuring is verifiable independently of the patched
library.

---

## 4. What changes, and what deliberately does not

### 4.1 The diff, in full

`config_ex4_multistep.yaml` inherits `config_ex2.yaml` and overrides **two lines**:

```yaml
neuron:
  spikingjelly:
    step_mode: m      # was s
    backend: cupy     # was torch
```

That is the whole scientific difference. Dataset, framing, T, batch size, optimiser,
learning rate, loss, epochs, seeds, metrics and **the neuron itself** — same tau, same
`decay_input`, same threshold, same reset, same surrogate — are inherited unchanged.

### 4.2 Why a new config file rather than editing ex2's

Three reasons, and the third is structural:

1. ex2's six completed runs must stay reproducible from the same commit.
2. The config hash differs, so results are traceable to which one produced them.
3. **`collect_results.py` refuses to merge rows from a different config.** So ex2 and
   ex4 cannot contaminate each other even by mistake. This is not vigilance; it is
   enforced.

### 4.3 Why SpikingJelly only

snnTorch and Norse have no equivalent fused kernel in this pipeline. Extending ex4 to
them would mean either writing sequence adapters that gain nothing (both would just
loop, like the control arm) or bringing in framework-specific optimisations that are
not the same change. Either would be a different experiment — see §9.

---

## 5. What you may and may not claim

**This section is the one to re-read before writing the report.** ex4's results are
easy to overstate.

### 5.1 ex4's speed numbers are NOT comparable to ex2's snnTorch and Norse rows

ex1 and ex2 rest on a guarantee: *identical architecture, identical pipeline, only the
framework changed.* ex4 breaks that guarantee deliberately — the execution strategy
changed too.

- ✅ "SpikingJelly trains N× faster in its fused-kernel mode **than SpikingJelly does
  in the mode ex1 and ex2 measured**."
- ❌ "SpikingJelly is N× faster than snnTorch." Not shown. snnTorch was never given a
  comparable opportunity.

The honest framing: ex4 establishes that **ex1's and ex2's SpikingJelly speed figures
were a lower bound, by a measured factor.** That is a strong, defensible claim and it
is enough.

### 5.2 The library is patched — say so

**[installed]** SpikingJelly 0.0.0.0.14 reads `np.int` at `auto_cuda/base.py:249`, an
alias numpy removed in 1.24. It is read on **every kernel launch**, so without a fix
the fused kernel cannot run at all.

It cannot be fixed by choosing a numpy version: Colab runs Python 3.12, whose earliest
numpy is 1.26, and the alias is gone from all of them. So
`src/multistep/sj_numpy_compat.py` corrects it at runtime.

Why the patch is safe to describe as changing nothing:

- The line is inside `check_ctypes`, a **validation** method. It asserts that an
  integer array was declared with an `int` type. It computes no arithmetic.
- `np.int` was literally an alias for Python's builtin `int`; numpy's own deprecation
  message says substituting `int` "will not modify any behavior and is safe".
- **The evidence is empirical, not just argued:** with the patch applied, spikes are
  bit-identical to the unpatched single-step path (§6).

This is a genuine finding worth reporting: a released framework whose advertised fast
path does not run against any currently-installable numpy. It is the **second** such
bug this project has caught — the first being Norse 1.1.0's SuperSpike ignoring its
`alpha` argument (`local_docs/norse_superspike_alpha_finding.md`). Two out of four
frameworks shipping a broken released feature is itself a result about the maturity of
the SNN framework ecosystem.

### 5.3 The kernel accelerates TRAINING only

**[installed]** `neuron.py:930` — `multi_step_forward` opens with `if self.training:`,
and the cupy branch is inside that block. In `eval()` mode it takes a `torch.jit` path
instead.

Consequences for the results table:

| Metric | Sees the fused kernel? |
|---|---|
| epoch time, training throughput | **yes** — this is the headline |
| accuracy | irrelevant, the maths is identical |
| **latency** (`measure_latency` calls `net.eval()`) | **no** — JIT path, not cupy |
| peak memory | changed, but see §5.4 |
| spike rates | identical by construction (§6) |

So do not describe the latency figure as "the fused kernel's latency". It may still
improve over ex2, because the JIT multi-step path beats our Python loop, but that is a
different mechanism and should be named as one.

### 5.4 Memory will move, and that is real

Folding T into the batch dimension changes the allocator's behaviour, so
`peak_memory_mb` will differ from ex2's. This is not a bug and not noise — it is a
genuine property of the execution mode, and the reserved-versus-allocated gap is
itself a framework characteristic (see `src/metrics.py`'s note on that). Report it as
a measured difference, not as an anomaly.

### 5.5 Do not quote 11×

SpikingJelly's paper claims ~11× training acceleration. That is a different network,
dataset and GPU. **[measured]** Ours is 4.99× on the neuron in isolation, and the
end-to-end figure will be **lower still** because Conv2d, MaxPool2d and Linear are
untouched and take time of their own.

Quote our own number, cite theirs as context, and state that the difference is
expected because only the neuron layers changed.

---

## 6. The equivalence check — built in, and free

Unlike ex2, ex4 has an unambiguous correctness criterion, and satisfying it costs
nothing extra.

**[measured]** `probe_spikingjelly_multistep.py --patch-spikingjelly`, Colab T4:

| Comparison | Result |
|---|---|
| spikes, fused kernel vs looped `'s'` | **max difference exactly 0** — bit-identical |
| gradients | 1.19e-07 absolute (≈ one float32 epsilon) |
| spike rate during the check | 21.9% — so the comparison was not vacuous |

**[measured]** Locally on CPU, the whole network (control arm vs ex2):

| Comparison | Result |
|---|---|
| weight fingerprint at seed 0 | identical (`d03b6a70b7398043`) |
| network output | `torch.equal` **True** at 17.5% spike rate, 410,758 spikes |
| input gradient through the network | agrees to 4.4e-11 |
| `spike_rate_pct`, `total_spikes`, `opportunities` | identical |
| `neurons` per layer | `[10800, 3872, 10]` in both modes |

### 6.1 Therefore: accuracy is the acceptance test

The neuron is mathematically the same, so **ex4's accuracy must land inside ex2's
~0.15 pp noise floor** against the ex2 SpikingJelly run at the same seed.

- Within the noise floor → the port is correct, and the speed number stands.
- Outside it → **the port is wrong.** Do not report it as a finding about the fused
  kernel; treat it as a bug.

ex2's completed seeds are therefore the acceptance criterion for ex4, at no extra
cost. That is unusually strong: most speed optimisations cannot be checked this way.

### 6.2 One trap that nearly produced a false pass

Worth recording because it would have been invisible. The first end-to-end comparison
passed while the network fired **zero spikes** — an untrained network at natural input
scale simply does not reach threshold, so `torch.equal` was comparing two tensors of
zeros. The figures in §6 are from a rerun at an input scale that gives a 17.5% firing
rate.

**General lesson for this project:** any equivalence check on a spiking network must
assert that spikes actually occurred. Both the probe and the CPU check now do. If a
future check reports suspiciously perfect agreement, check the spike count first.

---

## 7. Environment — the part that will bite you again

### 7.1 cupy and tonic disagree about numpy

| Package | Requires |
|---|---|
| `cupy-cuda12x` 14.x | numpy **≥ 2** |
| `tonic` 1.6.0 | numpy **< 2** |

And tonic is not optional: `train.py:48` imports `src.data`, which imports tonic at
module level (`data.py:41`). So both must work in one environment.

**Resolution:** `pip install "cupy-cuda12x<14"`. cupy 13.x is built against numpy 1.x.

**[measured]** Verified working on Colab (torch 2.11.0+cu128, CUDA 12.8, T4):

```
cupy-cuda12x 13.6.0  +  numpy 1.26.4  +  tonic 1.6.0
```

Install order matters: **`requirements.txt` first, cupy after.** `requirements.txt`
downgrades numpy for tonic, which would break a cupy already installed.

### 7.2 A missing cupy fails silently and late

**[installed]** `neuron.py:12-20` wraps `import cupy` in `except BaseException`,
logs at `info` level (below the default threshold, so nothing prints), and sets
`cupy = None`. A missing package, a version mismatch and a broken driver are
indistinguishable from outside, and **none of them raise at import** — the failure
surfaces as an `AttributeError` on `None` partway into training, after the data has
loaded and the clock has started.

`src/multistep/` therefore validates CUDA and cupy at **network-build time**. Run
`probe_spikingjelly_multistep.py` first regardless; it is a gate and takes a minute.

---

## 8. Concrete plan

### 8.1 Run order

```python
DRIVE = "/content/drive/MyDrive/snn_results"
CACHE = "/content/drive/MyDrive/snn_cache"
EXP   = "ex4"

# ── 0. environment. cupy AFTER requirements.txt (§7.1)
!pip install -q -r requirements.txt
!pip install -q "cupy-cuda12x<14"
!python -c "import numpy, tonic, cupy; print(numpy.__version__, tonic.__version__, cupy.__version__)"
!python check_env.py

# ── 1. THE GATE. Do not skip: it is the only thing that confirms the kernel
#       compiles for our parameters and agrees numerically (§6).
!python probe_spikingjelly_multistep.py --config config/config_ex4_multistep.yaml \
    --experiment ex4 --patch-spikingjelly

# ── 2. confirm the network is what you think it is. Expect backend 'cupy' and
#       three SpikingJellyMultiStepLIF layers.
!python check_network.py --config config/config_ex4_multistep.yaml --framework spikingjelly

# ── 3. data. Same cache as ex1/ex2 -- keyed by dataset settings, not experiment,
#       and ex4 changes none of them. Restores in seconds if archived.
!python prepare_data.py --config config/config_ex4_multistep.yaml --cache-archive $CACHE

# ── 4. FAST arm, 3 seeds
!for s in 0 1 2; do python train.py --config config/config_ex4_multistep.yaml \
    --experiment ex4 --framework spikingjelly --seed $s --results-root $DRIVE; done

# ── 5. CONTROL arm, same 3 seeds, same --experiment so both land together (§3)
!for s in 0 1 2; do python train.py --config config/config_ex4_control.yaml \
    --experiment ex4 --framework spikingjelly --seed $s --results-root $DRIVE; done
```

Then home, and merge as usual:

```
python collect_results.py --from <downloaded> --experiment ex4
```

`equivalence_check.py` is **not** in this list, and that is deliberate: it traces one
neuron one timestep at a time, so it is single-step by nature and refuses an `'m'`
config loudly. ex2's equivalence output already covers this neuron — ex4 does not
change the neuron, only how it is executed.

### 8.2 What the existing tooling gives for free

No new analysis code is needed. `runs.csv`, `epochs.csv` and `layers.csv` gain rows
with the same columns; the config hash distinguishes the two arms; the plotting
schema's families work unchanged because the metric names are unchanged.

### 8.3 Deliverables

- `experiments/ex4/results/` — 6 runs (3 seeds × 2 arms)
- `experiments/ex4/figures/` — speed comparison across the three arms
- `experiments/ex4/report_ex4.md` — same structure and style as `report_ex1.md`
- an amendment note in the ex1 and ex2 reports: **L2 is now quantified**, with the
  factor and a pointer here

### 8.4 The results table to aim for

| Arm | epoch time | accuracy | speedup vs ex2 | notes |
|---|---|---|---|---|
| ex2 baseline `s`+torch | *(have it)* | *(have it)* | 1.00× | from ex2 |
| ex4 control `m`+torch | | | expect ≈1.00× | unpatched, no cupy |
| ex4 fast `m`+cupy | | | **the headline** | patched, cupy 13.6.0 |

Accuracy column: all three must agree within ~0.15 pp. That row **is** the
correctness proof, so show it rather than asserting equivalence in prose.

---

## 9. After ex4 — the experiment this suggests

The obvious follow-up, recorded so the thought is not lost.

**Every framework in its own best available mode.** ex4 asks "how fast can
SpikingJelly go". The natural next question is "how fast can each of them go", which
is arguably the question a practitioner actually has — ex1 and ex2 both answer "how do
they compare when constrained to a common pipeline", which is the *scientific*
question but not the *practical* one.

The groundwork exists: the sequence network in `src/multistep/sequence_net.py` is
framework-agnostic, and the fold-into-batch trick applies to any framework. snnTorch
and Norse would need sequence adapters, and both would land near 1.00× (like the
control arm) unless they have optimisations we have not looked for.

That would be a fair "best mode per framework" comparison **and** it would isolate how
much of a framework's advantage is its kernel engineering rather than its neuron
model. Worth adding to `docs/open_items.md` if it appeals.

---

## 10. Sources

### Verified against the installed package
- **[installed]** `spikingjelly 0.0.0.0.14` — `neuron.py` (`multi_step_forward` 930,
  the `if self.training` branch 930, `supported_backends` 715, the swallowed cupy
  import 12-20), `base.py` (`MemoryModule.forward` 266, `multi_step_forward` 235),
  `auto_cuda/base.py` (`check_ctypes` 218, the `np.int` line 249, the every-launch
  call site 326), `functional.py` (`seq_to_ann_forward` 653), `surrogate.py`
  (`SurrogateFunctionBase` 118).

### Our own measurements
- **[measured]** `probe_spikingjelly_multistep.py`, Colab T4, 2026-08-12 — the 4.99×,
  the bit-identical spikes, the 1.19e-07 gradient agreement, the mode-B null result.
- **[measured]** Local CPU verification, 2026-08-12 — the weight fingerprint match,
  the whole-network output identity, the spike-metric identity.
- `experiments/ex1/report_ex1.md` — the ~0.15 pp accuracy noise floor (three agreeing
  estimates) used as ex4's acceptance criterion, and limitation **L2**.
- `experiments/ex2/ex2_design.md` §6 and §8.5 — where this follow-up was queued, and
  the L2 row it closes. Note §8.5 predicted this would need `SeqToANNContainer` and a
  network rewrite; both turned out to be unnecessary (see the companion document §3).

### Publications
- Fang et al., *Incorporating Learnable Membrane Time Constant to Enhance Learning of
  Spiking Neural Networks*, ICCV 2021 — the ~11× training acceleration claim, from the
  group that wrote SpikingJelly. Context only; do not quote as our result (§5.5).

### Engineering companion
- `local_docs/spikingjelly_multistep_intro.md` — the isolation design, the five silent
  failure modes, the full execution flow, and the verification logs.
