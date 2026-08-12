# Experiment 2 — Each framework out of the box, on N-MNIST

Comparing **snnTorch, SpikingJelly and Norse** by running the *same* network on all
three and letting each one use **its own default neuron**.

Run 2026-08-06 · Tesla T4 · **seeds 0, 1, 2 — 9 runs** · 5 epochs · one continuous session

**Experiment 1 is the control.** Same data, same architecture, same weights, same
optimiser, same seeds — the *only* thing that moved is the neuron's parameters.
Read the two reports as a pair.

---

## 1. Summary

All figures are **mean ± std over three seeds**, exactly as in Experiment 1.

- **Accuracy now separates, and by a lot.** 98.46 / 97.85 / 95.36 %. The gap between
  best and worst is **3.10 pp — about twenty times Experiment 1's ~0.15 pp noise
  floor**, and the ordering snnTorch > SpikingJelly > Norse is unanimous across all
  three seeds. Experiment 1 found no accuracy difference at all (0.06 pp). **The
  difference a user actually experiences between these libraries is a default, not
  an implementation.**
- **Only Norse's numbers moved outside its own noise.** Against Experiment 1's
  matched seeds: snnTorch **+0.07 pp** (inside the noise floor — snnTorch's result
  did not care which of the two neurons it got), SpikingJelly **−0.59 pp**, Norse
  **−3.02 pp**.
- **Norse had not finished training.** Its accuracy was still climbing steeply at
  epoch 5 (89.3 → 92.8 → 94.2 → 94.6 → 95.4 %) and its final training loss is
  0.175 against 0.044 / 0.059 for the others. Its shortfall is **at least partly
  "not converged in 5 epochs"**, not a settled ceiling (§7.1).
- **The headline prediction was wrong, and that is a result.** snnTorch's default
  neuron is twice as excitable as the other two (DC gain 2.0 vs 1.0), so it was
  predicted to spike most. It came **second**: SpikingJelly 5.58 %, snnTorch 4.34 %,
  Norse 3.49 %, unanimous across seeds. Excitability measured on an isolated neuron
  does not survive training — the network simply learns smaller weights (§7.2).
- **Everyone spikes roughly twice as much out of the box** — 4.34 / 5.58 / 3.49 %
  against Experiment 1's 2.57 / 2.66 / 2.21 %.
- **Peak memory inverted.** snnTorch went from **worst (901 MB) to best (470 MB)**,
  SpikingJelly 752 → 616 MB, Norse essentially unchanged at 763 MB. Still std
  exactly 0.0 MB in every case. This is the largest single change in the whole
  experiment and it is caused by the reset settings, not by the frameworks (§7.3).
- **Speed rankings mostly held, but two of them weakened.** SpikingJelly still has
  the best latency by a clear margin. But snnTorch vs Norse on latency, which did not
  overlap at all in Experiment 1, now overlaps — and SpikingJelly is no longer the
  fastest to train in every seed.
- **Energy is again reported, not interpreted** (§5.3). Every dynamic figure fell
  while spike rates doubled; §5.3 shows that for two of the three frameworks the
  entire drop is the idle baseline moving, not the work changing.
- **The neurons were verified to differ before any benchmarking** — worst membrane
  disagreement **0.786**, against Experiment 1's `1.2e-07`. Same script, same plot,
  opposite message.
- **Norse ran on a surrogate gradient with a released bug** (§4.3). Its `alpha` is
  silently ignored, and a user following the defaults inherits that. This is measured
  and shown, but it **cannot be separated from Norse's other default changes in this
  experiment** — the run that would separate them was not done (§8, L3).

| | snnTorch | SpikingJelly | Norse |
|---|---|---|---|
| accuracy % | **98.46 ± 0.04** | 97.85 ± 0.23 | 95.36 ± 0.25 |
| training time s | 590.6 ± 35.1 | **562.0 ± 5.5** | 586.1 ± 19.5 |
| throughput /s | 659.6 ± 8.4 | **718.2 ± 1.9** | 653.3 ± 51.2 |
| latency bs=1, ms | 33.05 ± 5.46 | **15.75 ± 2.78** | 27.92 ± 4.36 |
| spike rate % | 4.34 ± 0.53 | 5.58 ± 0.37 | **3.49 ± 0.25** |
| peak memory MB | **469.8 ± 0.0** | 616.0 ± 0.0 | 763.1 ± 0.0 |
| dynamic energy kJ | 7.5 ± 0.3 | 7.8 ± 0.3 | 9.9 ± 0.5 |

*Accuracy is bolded here, unlike in Experiment 1 — this time there is a winner, and
§7.1 explains what it is a winner at.* Energy is greyed out in intent, not
formatting: see §5.3.

---

## 2. What this experiment is

### The question

Experiment 1 forced all three libraries to compute **one** neuron, so that only the
implementation could differ. It found they were interchangeable: 0.06 pp apart on
accuracy, inside the noise.

That answers "are these libraries equivalent?" It does not answer the question a
user actually faces, which is:

> **If you install a framework and follow its own defaults and its own documented
> example for event-camera data, what neuron do you get — and what does it cost you?**

Nobody starts by harmonising three libraries. They start by typing `snn.Leaky(...)`,
`LIFNode()`, `LIFBoxCell()`. Experiment 2 measures what that gets you.

### What we did

1. Read each library's **installed source** and its **own documented example** for
   event data, and recorded the out-of-the-box neuron parameters (full derivation in
   [ex2_design.md](ex2_design.md) §2).
2. Put all three on **one common scale** — decay, input gain, DC gain, time constant
   — so libraries with different parameter names could be compared (§3).
3. **Measured the divergence** with the same equivalence script Experiment 1 used to
   measure agreement (§4).
4. Changed **only the neuron block** of the config. Network, data, weights,
   optimiser, learning rate, loss, timesteps, batch size and seeds are byte-identical
   to Experiment 1.
5. Trained 3 frameworks × 3 seeds and measured the same metrics.

### What can therefore differ

The library's implementation **and** its default neuron. That is one more free
variable than Experiment 1 had, which is why Experiment 1 has to be read alongside
this: it is the measurement that tells you the implementation contributes ~nothing,
so anything that moves here is the neuron.

**Network:** `12C5 – MP2 – 32C5 – MP2 – FC10`, 18,254 parameters, 20 timesteps.
**Data:** N-MNIST — handwritten digits recorded with an event camera, 60,000 train /
10,000 test, 34×34 pixels, 2 polarities.

### What deliberately did *not* change

Not each framework's whole recipe — only its neuron. Adopting each library's full
example would mean different datasets, batch sizes, optimisers, losses and
architectures, and then no metric would mean the same thing in two rows. The
reasoning is set out in [ex2_design.md](ex2_design.md) §5.2.

---

## 3. What changed: the three neurons on one scale

A leaky integrate-and-fire neuron holds one number, the membrane potential `v`, and
each timestep it leaks, integrates, then fires and resets:

```
v[t] = d · v[t-1]  +  g · x[t]          then, if v > θ:  spike, and reset
```

Two numbers describe the sub-threshold behaviour — the **decay `d`** (how much it
remembers) and the **input gain `g`** (how much of the input lands). Two more are
derived from them and are what actually make different libraries comparable:

- **τ = −1 / ln(d)** — how many timesteps the neuron remembers for.
- **DC gain = g / (1 − d)** — how high the membrane settles under a steady input,
  i.e. **how excitable the neuron is**. This is the one most easily missed, because
  it depends on `d` and `g` *together*.

**Measured**, not calculated on paper — each neuron was built with its out-of-the-box
arguments, its threshold raised out of reach to isolate the linear filter, and driven
with a constant input:

| | snnTorch | SpikingJelly | Norse |
|---|---|---|---|
| written as | `beta=0.5` | `tau=2.0`, `decay_input=True` | `dt=0.001`, `tau_mem_inv=100` |
| **decay `d`** | 0.5000 | 0.5000 | 0.9000 |
| **input gain `g`** | 1.0000 | 0.5000 | 0.1000 |
| **DC gain** | **2.0000** | 1.0000 | 0.9998 |
| **τ (timesteps)** | 1.44 | 1.44 | 9.49 |
| reset | **soft**, delayed 1 step | hard | hard |
| surrogate | atan(2.0) | ATan(2.0) | super — effectively α = 1 |

Three things fall out:

1. **Two of the three agree on decay, by coincidence.** snnTorch's tutorial β = 0.5
   and SpikingJelly's default τ = 2.0 both give d = 0.5. Nobody coordinated this;
   `1 − 1/2` simply lands on the same number.
2. **snnTorch is twice as excitable.** DC gain 2.0 against 1.0, because its gain is
   decoupled from its decay while the other two are normalised.
3. **Norse remembers ~6.6× longer** — τ = 9.49 steps against 1.44 — but each
   individual event lands with only a tenth of the strength.

### Against Experiment 1

| | ex1 (forced) | ex2 (out of the box) |
|---|---|---|
| snnTorch | β 0.9, hard reset, no delay, atan(2.0) | β **0.5**, **soft** reset, **delayed**, atan(2.0) |
| SpikingJelly | τ 10.0, `decay_input=False`, `detach_reset=False`, ATan(2.0) | τ **2.0**, `decay_input` → **True**, `detach_reset` → **True**, ATan(2.0) |
| Norse | dt 0.001, τ⁻¹ 100, `input_scale` = **10.0**, **circ(0.5)** | dt 0.001, τ⁻¹ 100, `input_scale` = **1.0**, **super** |
| **all three** | d = 0.9, g = 1.0, DC gain 1.0 | d, g and DC gain all differ |

Note what this means for Norse: Experiment 1 reached the common point only by adding
an external `input_scale = 10.0` — a factor **outside** the neuron, added because
Norse's `g` and `d` are locked together (`g = 1 − d`) and cannot be set
independently. Experiment 2 removes that scaffolding, which is the point, but it also
means **two things changed for Norse at once** — the gain and the surrogate. §8 (L3)
is explicit that this experiment cannot separate them.

---

## 4. Verification: the neurons now differ, and by how much

Experiment 1 ran `equivalence_check.py` to prove the neurons matched. Experiment 2
runs the **same script on the same two inputs** to measure how far apart they are.
The script issues no pass/fail, deliberately: ex1 wants agreement and ex2 expects
divergence, so no single threshold serves both. It reports the numbers and plots.

### 4.1 Result

| test | input | worst membrane disagreement | ex1's figure |
|---|---|---|---|
| constant step | 0.15 held from t=10 | **0.2297** | 1.19e-07 |
| Poisson | random, rate 0.15, amp 0.6 | **0.7860** | 1.19e-07 |

**Six to seven orders of magnitude apart.** In Experiment 1 the three traces agreed
to the last bit a 32-bit float can hold. Here the worst pair disagrees by 0.786 on a
membrane whose threshold is 1.0 — the disagreement is most of the operating range.

Peak membrane reached under each input, which is where the §3 table becomes visible
in a measurement:

| | snnTorch | SpikingJelly | Norse |
|---|---|---|---|
| constant step, peak `v` | **0.3000** | 0.1500 | 0.1500 |
| Poisson, peak `v` | **0.9012** | 0.4506 | 0.1572 |

The constant-step row is the DC-gain table read back out of the simulator: input
0.15 × DC gain 2.0 / 1.0 / 1.0 gives exactly 0.30 / 0.15 / 0.15.

The Poisson row shows something the DC-gain number alone does not. Norse reaches only
0.157 — far below the 0.6 its DC gain would suggest — because DC gain is the *steady
state*, and these inputs are isolated one-timestep events. Norse's per-event gain is
0.1, so a brief pulse barely moves it; it needs sustained input to accumulate. **For
event-camera data, which is sparse and bursty by nature, Norse's effective drive is
much weaker than its DC gain implies.** That is the most likely mechanical reason for
its low spike rate and slow learning in §5.

### 4.2 The Poisson test

![LIF membrane traces, ex2, Poisson input](equivalence/equivalence_poisson_20260806_214435.png)

Reading the figure, side by side with [Experiment 1's](../ex1/equivalence/equivalence_poisson_20260801_212009.png):

- **Top:** the input. Identical to Experiment 1's — same seed, same 14 events.
- **Middle:** the membrane voltages. In Experiment 1 *"three curves are plotted; you
  see one."* Here you see three. snnTorch (blue) peaks at 0.90, SpikingJelly (red
  dashes) at exactly half that, Norse (green dots) crawls along the bottom, never
  clearing 0.16 but never fully draining between events either — its long τ visible
  as the slow rising floor.
- **Bottom:** spike times. **All three are empty.** With the out-of-the-box neurons
  none of them reaches threshold on this input, where Experiment 1's forced neuron
  fired four times in all three. The reported "spike times match = 100 %" is
  therefore trivially true — 0 = 0 = 0 — and must not be read as agreement.

The constant-input test and the raw data are in [equivalence/](equivalence/).

### 4.3 The surrogate gradient, and Norse's `alpha`

> **What a surrogate gradient is.** A spike is a step function: it jumps from 0 to 1
> the instant the membrane crosses threshold. Its derivative is zero everywhere and
> infinite at one point, so backpropagation — which needs a usable derivative to
> decide how to correct each weight — cannot work through it. The fix is to keep the
> real step in the forward pass but **substitute a smooth bump for its derivative in
> the backward pass**. That substitute is the *surrogate gradient*, and every one of
> these three frameworks needs one.
>
> **What `alpha` is.** The surrogate's sharpness dial. A high `alpha` makes a narrow,
> tall bump — only neurons very close to threshold get corrected. A low `alpha` makes
> a wide, flat bump — neurons far from threshold still get corrected, and get
> corrected harder. Since most neurons sit far from threshold most of the time, alpha
> effectively sets **how big a learning step the whole network takes** at a given
> learning rate.

Norse's default surrogate is `method='super'` (SuperSpike) with `alpha=100.0`, which
reads like a very sharp, conservative choice. **It is not, because `alpha` is never
read.**

![Norse SuperSpike ignores its alpha](figures/FX.1_norse_superspike_alpha.png)

Panels A and C are **measured** — a tensor is pushed through the real surrogate on
the installed `norse 1.1.0` and `.backward()` is called — so no hand-copied formula
can make the figure disagree with the shipped code. Panel B is the published form,
which is the one place a formula is needed, because the shipped code cannot produce
it.

| panel | shows |
|---|---|
| **A** | the shipped surrogate at α = 1, 10, 100. Three curves are drawn; you see one. Largest difference between any two: **0.00e+00** — not small, *exactly zero*. |
| **B** | the same three α through the published formula `1/(α·\|x\|+1)²`. Largest difference: **0.88**. α is meant to matter across almost the whole range. |
| **C** | the shipped default against `atan(2.0)` (what the other two use) and `circ(0.5)` (what Experiment 1 substituted). |

The backward pass is `grad_output / (|x| + 1)²` with `alpha` never read, so it always
behaves as α = 1 — the **widest, strongest-gradient** setting available, the opposite
of what `alpha=100` suggests. Fixed upstream in commit `1d2671a` (Aug 2024),
**never released**; v1.1.0 remains current on PyPI.

Panel C carries the number that matters. All three surrogates agree *at* threshold
and diverge in the **tails**:

| \|v − θ\| | super, shipped | atan(2.0) | circ(0.5) | super ÷ atan |
|---|---|---|---|---|
| 0.0 | 1.0000 | 1.0000 | 1.0000 | 1.00× |
| 0.5 | 0.4444 | 0.2884 | 0.3536 | 1.54× |
| 1.0 | 0.2500 | 0.0920 | 0.0894 | **2.72×** |
| 2.0 | 0.1111 | 0.0247 | 0.0143 | **4.50×** |

Most neurons sit far from threshold most of the time, which is why this tail
difference compounded into the **6.03×** gradient-norm ratio Experiment 1 measured on
the real network's conv1 over 8 batches. At a shared learning rate of 2e-3, Norse is
effectively taking much larger steps than the other two.

**Why this is in the report rather than a footnote.** Experiment 1 deliberately
avoided the bug by substituting `circ(0.5)`. Experiment 2 must not avoid it: a user
who follows the defaults gets the buggy surrogate, so that *is* the out-of-the-box
condition. The bug stops being a caveat and becomes part of the measurement.

**What is not claimed.** That the surrogate bug caused Norse's 3.10 pp shortfall.
Norse's gain also dropped 10× between the experiments, and the two effects are
confounded here. See §8 (L3) and §6 (H6).

### 4.4 Network level — and a control across both experiments

| check | result |
|---|---|
| weight fingerprint (sha256 over all trainable parameters) | **identical for all three frameworks within every seed** — `30b4902cd20a39fe` (seed 0), `b80dbce09b1ebb09` (seed 1), `5f2de8c28fc112b2` (seed 2) |
| **the same three fingerprints as Experiment 1** | yes — byte-for-byte |
| trainable parameters | 18,254, identical (the neuron layers contribute none) |
| config hash | `20af039c9d78` for **all nine runs** |

The second row is worth more than it looks. Experiment 1 and Experiment 2 started
from **byte-identical weights within each seed**, so the ex1→ex2 comparisons in this
report are paired at the weight level, not merely at the seed label. Nothing in the
change of accuracy comes from one experiment getting a luckier initialisation.

The single config hash is also a change from Experiment 1, which spanned three (a
bookkeeping artefact documented there as L10). Here all nine runs were produced by
one file, `config/config_ex2.yaml`, in one session.

---

## 5. Results

Every metric opens with a short **"what this is"** block. Where that text would
simply repeat Experiment 1's, it is compressed and cross-referenced rather than
restated — Experiment 1 §5 is the fuller version, and nothing about the instruments
changed.

Each metric is given twice: **mean ± std over three seeds**, then **the three
individual seeds**, so nothing hides behind an average.

**One methodological difference from Experiment 1 that affects how the ± should be
read.** Experiment 1's nine runs were spread over several Colab sessions and several
accounts; Experiment 2's nine ran back-to-back in **one continuous ~2 h session**
(22:05 → 23:56 on 2026-08-06). Session-to-session drift is therefore largely absent
from ex2's timing stds. **Do not compare an ex1 timing std against an ex2 timing std
and conclude a framework became more consistent** — some of that is the session, not
the framework.

### 5.1 Accuracy

> **What this is.** Out of the 10,000 test digits the network never trained on, how
> many it labelled correctly. **"pp" = percentage points** — the plain gap between
> two percentages. See Experiment 1 §5.1 for the fuller version.
>
> **What is different this time.** In Experiment 1, accuracy was a *fairness check*:
> all three were computing the same network, so they had to land in the same place,
> and they did. Here they are computing genuinely different neurons, so for the first
> time a real accuracy difference is possible — and Experiment 1's noise floor is
> what makes it testable.

**Final accuracy, mean ± std over 3 seeds:**

| | snnTorch | SpikingJelly | Norse |
|---|---|---|---|
| accuracy % | **98.46 ± 0.04** | 97.85 ± 0.23 | 95.36 ± 0.25 |

**Per seed:**

| | seed 0 | seed 1 | seed 2 | spread |
|---|---|---|---|---|
| snnTorch | 98.50 | 98.42 | 98.46 | 0.08 |
| SpikingJelly | 97.62 | 97.84 | 98.08 | 0.46 |
| Norse | 95.61 | 95.35 | 95.12 | 0.49 |
| **spread across frameworks** | **2.89** | **3.07** | **3.34** | |

Read that bottom row against Experiment 1's, which was 0.08 / 0.10 / 0.05 pp. **The
framework-to-framework spread grew by a factor of about forty, while nothing about
the frameworks changed — only their default parameters did.**

The ordering `snnTorch > SpikingJelly > Norse` is **unanimous in all three seeds**,
and every pairwise gap clears Experiment 1's ~0.15 pp noise floor comfortably:

| pair | gap | vs noise floor |
|---|---|---|
| snnTorch − SpikingJelly | 0.61 pp | ~4× — **real** |
| SpikingJelly − Norse | 2.49 pp | ~17× — **real** |
| snnTorch − Norse | 3.10 pp | ~21× — **real** |

**Against Experiment 1, same seeds, same starting weights:**

| | ex1 | ex2 | change |
|---|---|---|---|
| snnTorch | 98.39 ± 0.14 | 98.46 ± 0.04 | **+0.07 pp — inside the noise floor** |
| SpikingJelly | 98.44 ± 0.17 | 97.85 ± 0.23 | **−0.59 pp** |
| Norse | 98.38 ± 0.20 | 95.36 ± 0.25 | **−3.02 pp** |

This is the cleanest result in the report. **snnTorch's default neuron is as good as
the carefully harmonised one** — its accuracy did not move at all, despite β dropping
from 0.9 to 0.5 and the reset switching from hard to soft-and-delayed. SpikingJelly
pays a real but modest 0.6 pp. Norse pays 3.0 pp.

**Accuracy per epoch, mean ± std over 3 seeds:**

| epoch | snnTorch | SpikingJelly | Norse |
|---|---|---|---|
| 1 | 97.17 ± 0.26 | 95.42 ± 0.80 | 89.26 ± 0.82 |
| 2 | 98.04 ± 0.16 | 97.29 ± 0.18 | 92.82 ± 0.04 |
| 3 | 98.23 ± 0.56 | 97.61 ± 0.11 | 94.17 ± 0.19 |
| 4 | 98.44 ± 0.21 | **98.09 ± 0.07** | 94.58 ± 0.27 |
| 5 | 98.46 ± 0.04 | 97.85 ± 0.23 | 95.36 ± 0.25 |

Two things to read here, and neither appears in the final-accuracy table:

**Norse never converged.** It is still gaining nearly a full point per epoch at the
end (94.58 → 95.36), and its curve has not flattened at any point. Compare Experiment
1, where all three were flat by epoch 4. Its final **training** loss says the same
thing: 0.175 against 0.044 (snnTorch) and 0.059 (SpikingJelly) — it has not even fit
the training set. **Norse's 95.36 % is not a ceiling; it is where five epochs
happened to stop.** Any statement of the form "Norse is 3 pp worse" must carry that
qualification.

**SpikingJelly's best epoch was not its last.** 98.09 % at epoch 4, 97.85 % at
epoch 5 — it went backwards. The drop (0.24 pp) is about the size of its own std, so
one epoch of wobble is the plausible reading rather than overfitting, but it is worth
recording that SpikingJelly's reported figure is not its best.

### 5.2 Speed

> **Three questions, three numbers.** *Training time* — a stopwatch over 5 epochs.
> *Throughput* — digits per second in batches of 128, the "factory" question.
> *Latency* — time to answer for **one** digit, the "queue" question. They are not
> each other flipped: at batch 1 most of the GPU sits idle and per-call Python
> overhead dominates. *Median* ignores freak outliers; *90th percentile*
> deliberately catches them. Experiment 1 §5.2 has the full explanation.

**Mean ± std over 3 seeds:**

| | snnTorch | SpikingJelly | Norse |
|---|---|---|---|
| training, 5 epochs | 590.6 ± 35.1 s | **562.0 ± 5.5 s** | 586.1 ± 19.5 s |
| per epoch | 118.1 ± 7.0 s | **112.4 ± 1.1 s** | 117.2 ± 3.9 s |
| throughput | 659.6 ± 8.4 /s | **718.2 ± 1.9 /s** | 653.3 ± 51.2 /s |
| latency, median | 33.05 ± 5.46 ms | **15.75 ± 2.78 ms** | 27.92 ± 4.36 ms |
| latency, 90th percentile | 40.30 ± 7.65 ms | **25.43 ± 4.40 ms** | 35.11 ± 7.81 ms |

**Per seed:**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **training time s** | | | |
| snnTorch | 590.8 | 625.6 | **555.4** |
| SpikingJelly | **555.7** | **564.5** | 565.8 |
| Norse | 589.5 | 603.8 | 565.2 |
| **throughput /s** | | | |
| snnTorch | 661 | 667 | **650** |
| SpikingJelly | **716** | **720** | **719** |
| Norse | 694 | 670 | 596 |
| **latency median ms** | | | |
| snnTorch | 30.50 | 39.32 | 29.32 |
| SpikingJelly | **18.95** | **13.94** | **14.35** |
| Norse | 32.94 | 25.05 | 25.77 |

Throughput and latency follow MLPerf's definitions [[1]](#references): throughput =
samples/s with batching (*Offline*), latency = one sample at a time (*Single-Stream*).

#### What holds and what does not

As in Experiment 1, two readings are possible — **across seeds** (compare the ± bands,
conservative) and **within a seed** (compare the three frameworks inside each seed and
check the ordering repeats, paired). Both are given.

**SpikingJelly's latency advantage holds decisively, under both readings.** Bands:
SpikingJelly 12.97–18.53, Norse 23.56–32.28, snnTorch 27.58–38.51 — SpikingJelly does
not touch either of the others, and it is fastest in all three seeds. Same conclusion
as Experiment 1, reached with different neurons.

**snnTorch vs Norse on latency collapses — and it did not in Experiment 1.** In
Experiment 1 the bands were 39.0–40.8 and 20.5–25.4, nowhere near touching, and
snnTorch was slower in 9/9 runs. Here the bands overlap (27.6–38.5 vs 23.6–32.3) and
the within-seed ordering is **not** unanimous: Norse is slower in seed 0 (32.94 vs
30.50), snnTorch is slower in seeds 1 and 2. **This is the one ranking that
Experiment 1 established and Experiment 2 overturns.** The mechanism is visible in
the numbers — snnTorch got faster (39.9 → 33.0 ms) while Norse got slower
(22.9 → 27.9 ms), and snnTorch's std widened from ±0.88 to ±5.46 ms. §7.3 connects
snnTorch's improvement to its much smaller memory footprint.

**Training time: SpikingJelly is still first, but no longer unanimously.** It is
fastest in seeds 0 and 1 but **second in seed 2**, where snnTorch's 555.4 s beat its
565.8 s. Across seeds the bands overlap heavily (556.5–567.5 vs 555.5–625.7). This is
weaker than Experiment 1, where the paired reading was 3-for-3. **Claim downgraded
from "holds" to "probable".**

What *is* unambiguous is SpikingJelly's **consistency**: ± 5.5 s on training time and
± 1.9 /s on throughput, against ± 35.1 s and ± 8.4 /s for snnTorch and ± 19.5 s and
± 51.2 /s for Norse. Some of that is the single-session setup helping everyone
equally, but the gap between SpikingJelly and the others within that same session is
real.

**Throughput: SpikingJelly clearly first; snnTorch vs Norse has no ranking.**
SpikingJelly beats both in all three seeds by 22–123 /s against a std of 1.9. snnTorch
(659.6 ± 8.4) and Norse (653.3 ± 51.2) are 6 /s apart with bands that almost entirely
overlap, and they swap places between seeds — Norse faster in seeds 0 and 1, snnTorch
faster in seed 2. Note Norse's ± 51.2: it declined 694 → 670 → 596 across the seeds
inside a single session, the largest unexplained drift in the experiment, and the
same monotone decline Experiment 1 saw (760 → 700 → 676) and also could not explain.
**Two independent experiments have now recorded Norse's throughput falling
monotonically with seed index.** That is worth chasing; it is recorded here, not
solved.

### 5.3 Energy

> **What this is.** How much electricity the GPU used while training, in joules — a
> total, not a rate. *Total* is everything the sensor saw. *Dynamic* is total minus
> what pure idling for the same duration would have cost, so it isolates the work
> from the cost of the GPU merely being switched on. The figure comes from the GPU's
> own sensor via NVIDIA's management library, polled at ~10 Hz. Experiment 1 §5.3 has
> the full treatment.

**Mean ± std over 3 seeds:**

| | snnTorch | SpikingJelly | Norse |
|---|---|---|---|
| total | 30.1 ± 1.8 kJ | 28.4 ± 1.5 kJ | 31.7 ± 1.6 kJ |
| dynamic | 7.5 ± 0.3 kJ | 7.8 ± 0.3 kJ | 9.9 ± 0.5 kJ |
| mean dynamic power | 11.05 ± 0.54 W | 12.25 ± 0.62 W | 14.52 ± 1.34 W |

**Per seed, dynamic energy (kJ):**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| snnTorch | 7.23 | 7.78 | 7.41 |
| SpikingJelly | 8.19 | 7.56 | 7.75 |
| Norse | 10.01 | 9.27 | 10.26 |

**Idle baselines, cold → hot (W):**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| snnTorch | 34.99 → 34.11 | 31.52 → 33.52 | 35.72 → 32.52 |
| SpikingJelly | 34.83 → 33.50 | 35.95 → 33.44 | 30.14 → 29.51 |
| Norse | 28.87 → 32.92 | 35.59 → 33.41 | 30.73 → 30.07 |

**No run carries a validity warning this time** — Experiment 1 had one (Norse seed 2's
unstable baseline). That is an improvement, and it is the only reason the
self-contradiction below is worth pointing at: it is not attributable to a flagged
bad run.

> ### ⚠ Energy is reported here, not interpreted — and this experiment sharpens why
>
> The stance is carried over unchanged from Experiment 1: these numbers are recorded,
> the agreed plots are built from them, and **no conclusion is drawn from energy**.
> Experiment 2 adds a fifth reason to Experiment 1's four, and it is the most concrete
> one yet.
>
> **Spike rates roughly doubled, and every dynamic energy figure fell.**
>
> | | spike rate ex1 → ex2 | dynamic energy ex1 → ex2 |
> |---|---|---|
> | snnTorch | 2.57 → 4.34 % (+69 %) | 9.67 → 7.47 kJ (**−23 %**) |
> | SpikingJelly | 2.66 → 5.58 % (+110 %) | 8.42 → 7.83 kJ (**−7 %**) |
> | Norse | 2.21 → 3.49 % (+58 %) | 13.09 → 9.85 kJ (**−25 %**) |
>
> On the face of it that says more spikes cost less energy. **It says no such thing,
> and the cause is traceable.** The dynamic figure is `total − (hot idle baseline ×
> duration)`, and the hot baseline moved between the two experiments:
>
> | | hot idle, ex1 | hot idle, ex2 | Δ | Δ × duration | actual dynamic drop |
> |---|---|---|---|---|---|
> | snnTorch | 30.10 W | 33.38 W | +3.28 W | **2.22 kJ** | 2.20 kJ |
> | SpikingJelly | 30.91 W | 32.15 W | +1.24 W | **0.79 kJ** | 0.59 kJ |
> | Norse | 30.61 W | 32.13 W | +1.52 W | **1.03 kJ** | 3.24 kJ |
>
> For snnTorch the baseline shift accounts for **essentially 100 %** of the apparent
> saving; for SpikingJelly it more than accounts for it. Their **total** energy barely
> moved at all (30.57 → 30.10 kJ and 27.87 → 28.40 kJ). **Nothing about the work
> changed; the subtrahend did.** Only Norse shows a drop that survives the correction
> — its total genuinely fell 34.97 → 31.74 kJ — and even that is confounded with it
> running 34 s shorter.
>
> Note also that the ex2 baselines are systematically 1–3 W higher than ex1's on the
> same GPU model, and that **cold → hot now usually falls** (7 of 9 runs) where in
> Experiment 1 it usually rose. A baseline that drifts by that much between sessions,
> in an inconsistent direction, is not a stable reference to subtract against.
>
> **Therefore:** treat §5.3 as raw instrument output for the record, as in Experiment
> 1. The action item is unchanged and now has a number attached: **pin down the idle
> baseline before energy carries any argument.**

### 5.4 Spiking activity

> **What this is.** How talkative the neurons are. Each digit is shown over 20
> timesteps, so every neuron gets 20 chances to fire; the spike rate is the share it
> used. 5 % means firing on one opportunity in twenty. On neuromorphic hardware a
> neuron that does not fire costs almost nothing, so **the same accuracy with fewer
> spikes is a genuine win**. Reported per layer as well as overall, because the big
> early layer dominates the average. **Layer 2 has only 10 neurons** and its
> percentages bounce around accordingly — see Experiment 1 §7.2.

**Mean ± std over 3 seeds:**

| | snnTorch | SpikingJelly | Norse |
|---|---|---|---|
| overall | 4.339 ± 0.530 % | 5.582 ± 0.369 % | **3.490 ± 0.249 %** |
| in spikes per neuron per digit | 0.868 | 1.116 | **0.698** |
| layer 0 (10,800 neurons) | 4.664 ± 0.731 % | 6.029 ± 0.505 % | **3.711 ± 0.301 %** |
| layer 1 (3,872 neurons) | 3.405 ± 0.330 % | 4.323 ± 0.051 % | **2.869 ± 0.403 %** |
| layer 2 (10 neurons) | 14.983 ± 1.930 % | 10.616 ± 0.200 % | **5.265 ± 0.162 %** |

**Per seed, overall spike rate (%):**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| snnTorch | 4.066 | 4.950 | 4.000 |
| SpikingJelly | 5.174 | 5.891 | 5.682 |
| Norse | **3.250** | **3.747** | **3.473** |

**Norse is sparsest again, unanimously** — as in Experiment 1, in every seed, with
completely different parameters. Its band (3.25–3.75) clears snnTorch's (3.81–4.87)
and SpikingJelly's (5.21–5.95). That is the only spiking result that replicates
across both experiments, and it now has two different mechanical explanations behind
it (§7.2).

**The prediction that failed.** From the DC-gain table in §3, snnTorch's neuron is
twice as excitable as the other two, and §4.1's isolated-neuron test confirms this
directly — at input 0.6 snnTorch reaches 0.90 where SpikingJelly reaches 0.45. The
written-in-advance hypothesis was therefore *"expect snnTorch's spike rate to be
markedly higher than the other two"*. It is **markedly lower than SpikingJelly's**,
in all three seeds. §7.2 works through why.

**Everyone got noisier out of the box:**

| | ex1 | ex2 | change |
|---|---|---|---|
| snnTorch | 2.567 % | 4.339 % | +69 % |
| SpikingJelly | 2.661 % | 5.582 % | +110 % |
| Norse | 2.211 % | 3.490 % | +58 % |

**Experiment 1's forced neuron is the sparser configuration for all three
frameworks** — at equal or better accuracy in every case. Whatever else the defaults
buy, they cost spikes, which is the currency this whole class of hardware is meant to
save.

**Sparsification during training** (measured on the training set at each epoch, which
runs slightly above the test-set figure quoted above):

| | epoch 1 | epoch 5 | change | ex1's change |
|---|---|---|---|---|
| snnTorch | 5.060 % | 4.587 % | −9 % | −16 % |
| SpikingJelly | 6.259 % | 5.983 % | −4 % | −14 % |
| Norse | 4.178 % | 3.730 % | −11 % | −17 % |

All three still find a quieter solution as they train, but **all three sparsify
noticeably less than in Experiment 1**. For Norse the likely reason is simply that it
is only a third of the way through learning (§5.1); for the other two it is
unexplained and recorded as an observation.

**The output layer is where snnTorch's excitability does show up.** Layer 2:
snnTorch 14.98 %, SpikingJelly 10.62 %, Norse 5.27 % — a nearly 3× spread and the one
layer where snnTorch is the loudest, matching the §3 prediction. Treat it with the
caution Experiment 1 established: 10 neurons, and snnTorch's ± 1.93 pp there is the
widest error bar in the report. It is a pointer, not a finding.

### 5.5 Memory

> **What this is.** The **high-water mark** — the most GPU memory in use at any single
> instant. *Allocated* is what the numbers occupied; *reserved* is what PyTorch
> claimed from the driver. Training needs far more than inference because the
> intermediate values from all 20 timesteps must be kept for the backward pass.
> Experiment 1 §5.5 has the full version.

**Mean ± std over 3 seeds:**

| | snnTorch | SpikingJelly | Norse |
|---|---|---|---|
| allocated, training | **469.8 ± 0.0 MB** | 616.0 ± 0.0 MB | 763.1 ± 0.0 MB |
| reserved, training | **538 ± 0 MB** | 678 ± 0 MB | 838 ± 0 MB |
| allocated, inference | 81.9 ± 0.0 MB | **68.8 ± 0.0 MB** | 91.4 ± 0.0 MB |

**Per seed:** every framework returned the identical figure in all three seeds, to the
byte, nine runs out of nine — as in Experiment 1. There is no per-seed table because
there is no variation. The zero std is expected, not suspicious: peak memory is
decided by the *shape* of the computation, which the random seed does not touch.

**This is the largest change between the two experiments, and the ranking inverted:**

| | ex1 | ex2 | change |
|---|---|---|---|
| snnTorch | 901.0 MB *(worst)* | **469.8 MB** *(best)* | **−431.2 MB, −48 %** |
| SpikingJelly | 752.4 MB *(best)* | 616.0 MB | −136.4 MB, −18 % |
| Norse | 760.7 MB | 763.1 MB | +2.4 MB, +0.3 % |

snnTorch went from needing 148.5 MB **more** than SpikingJelly to needing 146.2 MB
**less**. Experiment 1 called its memory gap *"the only difference in this report with
no uncertainty attached at all"* — which was true of that configuration, and is the
point: **a zero-variance measurement can still be a property of the settings rather
than of the library.** Both experiments measure exactly what they measure, and
neither licenses "snnTorch uses more memory than SpikingJelly" as a statement about
the libraries.

§7.3 sets out the mechanism, which is the reset settings, and is explicit that it is
an explanation offered rather than an ablation performed.

### 5.6 The story the three seeds tell

> **Seed.** The number the random-weight generator and the data shuffler start from.
> Same seed → the same "random" weights and the same shuffle. **Mean** is the average
> of the three runs; **std** is how spread out they were. Experiment 1 §5.6 explains
> at length why three of them are the minimum honest number, and that reasoning is not
> repeated here — it did not become invalid because the neuron changed.

What the seeds bought this time is different from what they bought in Experiment 1.
There, the seeds' job was to **demolish** apparent findings: three of nine
single-seed claims did not survive. Here, most of the effects are so far outside the
noise that the seeds mainly **confirm** them — but they still did three jobs:

**1. They made the accuracy result unarguable.** The framework ordering is unanimous
3/3 and the smallest gap (0.61 pp) is four times the noise floor. A single seed would
have shown the same ordering, but could not have shown that seed 2 nearly closed the
snnTorch–SpikingJelly gap (98.46 vs 98.08, 0.38 pp) while seed 0 opened it wide
(98.50 vs 97.62, 0.88 pp). **The size of that particular gap is seed-dependent even
though its sign is not.**

**2. They killed the one new speed ranking.** snnTorch vs Norse on latency looks
decisive in seed 1 (39.32 vs 25.05 ms — snnTorch 57 % slower) and reverses in seed 0
(30.50 vs 32.94 — Norse slower). One seed would have published a false ranking here,
exactly as Experiment 1 warned. It is also the ranking Experiment 1 itself had
established with three seeds — so **the lesson is not just "use three seeds" but "a
ranking is a property of a configuration, not of a library"**.

**3. They exposed that Norse's throughput decline replicates.** 694 → 670 → 596 /s
across seeds 0, 1, 2, in one session — and Experiment 1 recorded 760 → 700 → 676 /s
across the same three seed labels, in different sessions. Two experiments, the same
monotone pattern, no explanation. A one-seed or two-seed run could not have surfaced
this at all.

#### What the ± is made of

Same two sources as Experiment 1 — the starting point, and the machine — and this
experiment still cannot separate them. But the mixture is different: ex2's nine runs
share one session, so the machine contribution to the *timing* stds is smaller than
ex1's, while the *accuracy* stds (0.04 / 0.23 / 0.25) straddle ex1's range
(0.14–0.20). snnTorch's 0.04 pp is the tightest accuracy std in either experiment and
SpikingJelly's and Norse's are the widest — so **the noise floor is not a constant of
the project; it depends on the configuration too.** Experiment 1's ~0.15 pp is used
throughout this report as the reference because it is the best-supported estimate
available (three agreeing methods), but it is a reference, not a law.

#### The fairness check, carried across experiments

`weight_fingerprint` behaves exactly as it must, and this time it also links the two
experiments (§4.4): identical across the three frameworks within each seed, different
between seeds, and **identical to Experiment 1's for the same seed**. Every
comparison in this report — within ex2, and ex1 vs ex2 — is paired at the level of
the actual starting weights.

---

## 6. The hypotheses, scored

Six predictions were written down in [ex2_design.md](ex2_design.md) §7 **before any
training happened**, so the results could contradict them. Three held, one held with
a correction, one was refuted, and one turned out to be untestable with the runs
performed.

| # | prediction | verdict |
|---|---|---|
| **H1** | The neurons diverge, and that divergence is the headline | **Holds.** Worst membrane deviation 0.786 against ex1's 1.19e-07 (§4.1), and accuracy separates by 3.10 pp against ex1's 0.06 pp (§5.1). |
| **H2** | snnTorch's spike rate is much higher than the other two, because its DC gain is 2.0 vs 1.0 | **Refuted.** SpikingJelly is highest at 5.58 %, snnTorch second at 4.34 %, unanimous in 3/3 seeds. The DC-gain prediction is correct about the *isolated neuron* (§4.1 measures exactly 2×) and wrong about the *trained network* (§7.2). |
| **H3** | Norse's activity is smoothest over time, snnTorch's burstiest | **Untested.** Per-timestep spike rates are still not recorded (metrics item M3). §4.2's membrane traces are consistent with it — Norse's trace never fully drains between events while the other two return to ~0 — but that is one neuron on synthetic input, not the network. |
| **H4** | Accuracy differences exceed ex1's ~0.15 pp noise floor | **Holds decisively.** All three pairwise gaps are 4–21× the noise floor (§5.1). |
| **H5** | Speed, memory and latency rankings roughly unchanged from ex1 | **Holds for speed, fails for memory.** SpikingJelly still leads latency and throughput. But peak memory **inverted** — snnTorch went from worst to best, a 431 MB swing (§5.5) — and snnTorch vs Norse on latency went from non-overlapping to overlapping (§5.2). The design note said *"if a ranking does move, that is interesting and needs explaining"*; §7.3 explains it. |
| **H6** | Norse is handicapped by the SuperSpike bug, not by Norse | **Cannot be decided from these runs.** The bug is real and measured (§4.3), and Norse's shortfall is real and measured (§5.1) — but between ex1 and ex2 Norse's surrogate *and* its input gain both changed (circ(0.5) → super, gain 1.0 → 0.1). The two are confounded. §8 (L3) names the single missing run that would settle it. |

**H2 and H6 are the two most valuable rows.** H2 is a prediction derived correctly
from the equations, verified on an isolated neuron, and then falsified by the trained
network — which is a specific, transferable lesson about what neuron-level analysis
can and cannot predict. H6 is a hypothesis the experiment was designed to test and
**did not**, because of a confound that was visible in the design (§3 flags that two
things change for Norse) and was not caught before running. Both are recorded as
such.

### Framework findings, carried over and added to

Experiment 1 listed nine framework-level issues found while establishing equality.
All nine still stand. Experiment 2 confirms one of them under the condition that
actually matters and adds two observations:

| # | finding | evidence |
|---|---|---|
| ex1 #1 | **Library defaults are not the same neuron** | Confirmed, and now quantified in accuracy: 3.10 pp (§5.1). In ex1 this was a source-read; here it is a measured cost. |
| ex1 #2–#4 | **Norse's SuperSpike ignores its `alpha`** | Re-verified on the installed norse 1.1.0 while producing §4.3's figure: α = 1, 10, 100 give byte-identical gradients. Still unreleased upstream. |
| **new** | **SpikingJelly's own event-data example contradicts its constructor defaults** — the example passes `ATan(2.0)` and `detach_reset=True`; the constructor defaults are `Sigmoid(4.0)` and `detach_reset=False` | Read from the installed `spikingjelly/activation_based/examples/classify_dvsg.py` vs `neuron.LIFNode.__init__`. This report follows the example; the bare-defaults variant was not run (§8, L4). |
| **new** | **snnTorch has no default `beta` at all** — the constructor requires it | Installed snntorch 1.0.0. The 0.5 used here is from its own N-MNIST tutorial, i.e. an example value standing in for a non-existent default (§8, L5). |

---

## 7. Discussion

### 7.1 What snnTorch actually won

snnTorch is 3.10 pp ahead of Norse and 0.61 pp ahead of SpikingJelly, and both gaps
are far outside the noise. It is worth being precise about what that does and does
not mean, because the sentence "snnTorch is the most accurate framework" is not
supported by these two experiments together — **Experiment 1 measured that claim
directly and found it false.**

What is supported: **snnTorch ships the default neuron that costs its users the
least.** Its out-of-the-box configuration lands within noise of the carefully
harmonised one (+0.07 pp), while SpikingJelly's costs 0.59 pp and Norse's costs
3.02 pp. That is a real, useful, user-facing property — but it is a property of the
defaults chosen, and any of the three could change it in a release.

There is a caveat that cuts specifically against reading Norse's number as a ceiling.
**Norse had not converged.** Its accuracy was still gaining ~0.8 pp per epoch at
epoch 5 and its training loss (0.175) is four times the other two's — it has not yet
fit the *training* set, let alone plateaued on the test set. A longer run would close
some unknown part of the 3.10 pp. This does not make the result wrong: five epochs
was the budget, held equal for all three, and "converges more slowly under a shared
recipe" is itself a cost a user pays. But the honest phrasing is **"Norse reaches
95.36 % in five epochs"**, not "Norse tops out at 95.36 %".

A likely mechanical contributor is in §4.1: Norse's per-event gain is 0.1, so on the
sparse, bursty input an event camera produces, its neurons are driven roughly ten
times more weakly than in Experiment 1. Weak drive means fewer spikes (confirmed:
sparsest in 3/3 seeds), and fewer spikes means less gradient signal reaching the
weights. Combined with a surrogate taking ~6× larger steps at the same learning rate
(§4.3), Norse is running a rather different optimisation problem from the other two
under a shared learning rate — which Experiment 1 already flagged as limitation L7 and
which this experiment makes worse rather than better.

### 7.2 Why the excitability prediction failed

H2 predicted snnTorch would spike most, from a DC gain of 2.0 against 1.0. §4.1
confirms the premise exactly — driven with the same input, snnTorch's isolated neuron
reaches precisely twice the membrane of the other two. And yet SpikingJelly ends up
firing 29 % more often than snnTorch across the trained network.

The reason is that **the prediction holds the weights fixed and training does not.**
A DC gain of 2.0 means each unit of synaptic input produces twice the membrane
response. Backpropagation sees exactly the same thing the prediction did, and does the
obvious: it learns smaller weights. The neuron's excitability is a constant the
optimiser can trivially absorb into the weight scale, because both sit in the same
product `g · W · x`. What the optimiser *cannot* absorb is anything that changes the
shape of the computation rather than its scale.

That reframes what the §3 table is for. **Decay τ and reset type are structural —
they change what the neuron computes over time. DC gain is a scale factor, and scale
factors are what gradient descent is best at cancelling.** The three quantities were
presented as equally important in the design; they are not, and the spike-rate result
is what shows it.

Two secondary observations support this reading rather than an alternative:

- **The output layer, where absorption is weakest, follows the prediction.** Layer 2's
  10 neurons feed directly into the loss and have the least room to rescale; there
  snnTorch *is* the loudest, 14.98 % vs 10.62 % vs 5.27 % (§5.4). Ten neurons is a
  weak instrument (Experiment 1 §7.2), so this is corroboration, not proof.
- **What SpikingJelly's high rate does track is its accuracy shortfall.** It fires
  most (5.58 %) and scores second (97.85 %), while Norse fires least (3.49 %) and
  scores last. Across these two experiments the spike rate does not predict accuracy
  in either direction — ex1's sparsest framework (Norse) was tied for most accurate.
  **Sparsity and accuracy are separate axes**, which is what makes the joint plot
  ([F3.2_tradeoff_pareto.png](figures/F3.2_tradeoff_pareto.png)) worth having.

Norse being sparsest in both experiments is the one spiking result that replicates —
but note it does so for **different reasons each time**: in Experiment 1 with an
identical gain and decay to the others, attributable to its surrogate (`circ(0.5)`);
here with a 10× weaker gain, attributable mostly to drive. The same observation, two
mechanisms. That is a reason to be careful about calling it "a property of Norse".

### 7.3 Why the memory ranking inverted

snnTorch dropped 431 MB (−48 %) and SpikingJelly 136 MB (−18 %), while Norse moved
2.4 MB. Since peak memory has zero seed variance and the network shape is unchanged,
the cause has to be in the neuron settings.

**The explanation offered, for snnTorch: hard reset stores more than soft reset.**
A hard reset is implemented as a multiplication, `mem = mem * (1 − spk)`; autograd
must keep **both** operands for every timestep to compute that product's backward
pass. A soft reset is a subtraction, `mem = mem − spk · θ`, whose backward needs no
saved tensors at all. With 20 timesteps and layer 0 alone holding
128 × 10,800 × 4 bytes ≈ 5.5 MB per tensor per step, retaining two extra tensors
across 20 steps is ~220 MB at that layer before layer 1 is counted. Experiment 1
forced `reset_mechanism="zero"` (hard) and `reset_delay=False`; Experiment 2 uses the
defaults, `"subtract"` (soft) and `reset_delay=True`. The order of magnitude fits the
observed 431 MB.

**For SpikingJelly: `detach_reset`.** It was `False` in Experiment 1 and is `True`
here, which is precisely a flag that stops gradient flowing through the reset path —
so the graph no longer needs to retain it. A 136 MB saving on the same network is
consistent with that.

**For Norse: nothing memory-relevant changed** — the reset stayed hard, and removing
`input_scale` removes one cheap multiply. +2.4 MB is consistent with the surrogate
swap saving slightly different tensors, and is not worth explaining further.

**This is an explanation, not a measurement.** No ablation was run isolating
`reset_mechanism`, `reset_delay` or `detach_reset` one at a time, and both of
snnTorch's reset settings changed together. The account above is consistent with
every number and with how autograd works, but it is a hypothesis, and it is listed in
§8 (L6) as the cheapest outstanding measurement in the project — three short runs, no
training required, since peak memory has zero variance.

The same mechanism plausibly explains snnTorch's latency improvement (39.9 → 33.0 ms,
§5.2): a graph with roughly half the retained tensors is less work to build and free
per call, and per-call overhead is exactly what dominates at batch size 1.

### 7.4 What the pair of experiments says together

Reading Experiment 1 and Experiment 2 as one study, three statements are supported
that neither supports alone:

**1. The libraries are interchangeable; their defaults are not.** Forced onto one
neuron, the three land within 0.06 pp — inside the noise. Left on their own defaults,
they spread 3.10 pp. Since Experiment 1 establishes that the implementations
contribute nothing measurable to accuracy, the entire ex2 spread is attributable to
parameter choices a user could change in one line.

**2. Speed and memory rankings are configuration-dependent, and only one of them
survived the change.** SpikingJelly's latency lead held under both neurons (and both
readings, in both experiments) — it is the single most robust performance claim in
the project. snnTorch vs Norse latency did not survive. Peak memory did not merely
fail to survive; it **inverted**. Any benchmark reporting a memory ranking without
stating its reset settings is reporting the settings, not the framework.

**3. The most consequential differences between these libraries are in what they
choose for you.** snnTorch's tutorial β and soft reset cost nothing and save half the
memory; SpikingJelly's τ = 2.0 with `decay_input=True` costs 0.6 pp and doubles the
spike count; Norse's locked gain and unreleased surrogate bug cost 3.0 pp and leave
it unconverged at the shared epoch budget. None of these are implementation quality.
All three are decisions in a default argument.

### 7.5 Against published work

| our result | published | reading |
|---|---|---|
| accuracy 95.36–98.46 % | 98.74–98.93 % (earlier SNNs), 99.28–99.40 % (recent optimised) [[6]](#references) | snnTorch's 98.46 % sits where Experiment 1's runs did, slightly below published work for the expected reasons (18K parameters, 5 epochs, 20 timesteps, no tuning). SpikingJelly and especially Norse fall well below, and this report attributes that to defaults and to non-convergence, not to the libraries. |
| 0.70–1.12 spikes per neuron per digit | *"about 0.4 spikes per neuron"* for convolutional SNNs [[4]](#references) | **Above the published range**, where Experiment 1's 0.44–0.53 was inside it. Out-of-the-box neurons are meaningfully noisier than the literature's tuned ones. |
| spike rate 3.49–5.58 % | *">75 % of neurons remain silent"* on N-MNIST; ~5 % exceed 10 % firing [[4]](#references) | Still consistent with a mostly-silent population, but closer to the reported upper end than Experiment 1 was. |

**Positioning.** The most recent multi-framework benchmark [[2]](#references) compares
SpikingJelly, BrainCog, Sinabs, SNNGrow and Lava, excludes snnTorch and Norse, and
does not verify that the frameworks compute the same neuron before comparing them.
This pair of experiments covers those two libraries, adds the verification step, and
adds the measurement that the verification step is *worth* 3.10 pp — because a
benchmark that skips it is measuring default parameters and reporting them as
framework quality.

---

## 8. Limitations

| # | limitation |
|---|---|
| L1 | **Three seeds bounds the noise, it does not characterise it.** Same caveat as Experiment 1: the ± rests on three points and mixes the starting point with the machine, and cannot separate them. |
| L2 | **ex2's nine runs share one continuous session** while ex1's spanned several sessions and accounts, so ex1 and ex2 *timing* stds are not directly comparable (§5). A framework that looks more consistent here may simply have been measured under less session drift. |
| **L3** | **Norse's confound — the most important limitation here.** Between ex1 and ex2, Norse's surrogate changed (`circ(0.5)` → `super`) *and* its input gain changed (1.0 → 0.1). Its 3.02 pp drop cannot be attributed between them, so **H6 is undecided**. The run that would settle it already exists as a config and was not executed: `config/norse_super.yaml` is ex1's neuron with only the surrogate swapped to `super`. One framework × three seeds, ~40 min. |
| L4 | **The bare-defaults variant (ex2b) was not run.** SpikingJelly's constructor defaults (`Sigmoid(4.0)`, `detach_reset=False`) disagree with its own event example (`ATan(2.0)`, `detach_reset=True`); this experiment follows the example, on the reasoning that a shipped example *is* what the framework tells you to do. `config/config_ex2_bare.yaml` was designed but never written or run, so **the measured cost of "read the docs" vs "accept the defaults" is still unknown.** |
| L5 | **snnTorch's `beta = 0.5` is an example value, not a default.** snnTorch requires `beta` and has no default at all, so the out-of-the-box condition for snnTorch is partly a choice we made. It is its own N-MNIST tutorial's value, which is the best available evidence, but it is not the same kind of number as SpikingJelly's `tau=2.0`. |
| L6 | **The memory explanation (§7.3) is not measured.** Both of snnTorch's reset settings changed together and no single-flag ablation was run. Cheapest outstanding measurement in the project — peak memory has zero seed variance, so one short run per flag settles it. |
| L7 | **Norse is not strictly out of the box.** It uses `LIFBoxCell`, carried over from Experiment 1, because Norse's headline `LIFCell` is a **second-order** neuron with an extra synaptic-current state that structurally does not match the other two. Switching would change the neuron's order, not just its parameters. |
| L8 | **Norse did not converge in 5 epochs** (§5.1, §7.1). Its result is a five-epoch figure, not a ceiling, and the shared epoch budget is part of the recipe rather than a property of the library. |
| L9 | **One shared learning rate** (2e-3), carried over from Experiment 1's L7 — and the risk is *larger* here, because the surrogate gradients now genuinely differ. A framework may be underperforming from gradient-scale mismatch rather than anything intrinsic. The measurement that would settle it is a per-framework gradient-norm probe, which `equivalence_check.py` already supports. |
| L10 | **SpikingJelly ran without its fused CUDA kernel** (`step_mode='s'`, `backend='torch'`), same as Experiment 1's L2. Its speed advantage is a **lower bound**. |
| L11 | **GPU energy is relative, not absolute**, and is deliberately left uninterpreted (§5.3). The sensor updates at ~10 Hz (measured 100.3–101.9 ms across all nine runs) and differs from a physical meter by up to 73 % [[5]](#references). Experiment 2 adds a specific failure: the idle baseline drifted 1–3 W between the two experiments and accounts for the entire apparent energy saving in two of three frameworks. |
| L12 | **Per-timestep spike rates are still not recorded**, so H3 could not be tested at the network level (metrics item M3). |
| L13 | **One dataset, one architecture, one shared cloud GPU.** N-MNIST is comparatively easy and all nine runs were on Colab T4s. Conclusions may not transfer to harder event datasets or larger networks. |

*Limitation numbers are not shared between reports — Experiment 1's L2 (SpikingJelly's
fused kernel) is L10 here, and its L7 (one shared learning rate) is L9.*

---

## 9. Reproduction

```bash
CFG=config/config_ex2.yaml               # each framework's own out-of-the-box neuron

.venv/Scripts/python probe_norse_alpha.py --experiment ex2   # §4.3 figure, local, no GPU

python check_env.py                                        # versions, GPU, power-sensor rate
python equivalence_check.py --config $CFG --experiment ex2  # §4 — the divergence IS the result
python check_network.py  --config $CFG --all                # identical weights across frameworks
python prepare_data.py   --config $CFG                      # build the shared cache FIRST

# then, for each seed in 0 1 2, all three frameworks:
python train.py --config $CFG --experiment ex2 --framework snntorch     --seed 0
python train.py --config $CFG --experiment ex2 --framework spikingjelly --seed 0
python train.py --config $CFG --experiment ex2 --framework norse        --seed 0
```

`--config` is required and has no default, so a run can never silently pick up another
experiment's neuron. **All nine runs record `config/config_ex2.yaml` and the single
config hash `20af039c9d78`** — unlike Experiment 1, which spanned three hashes for
bookkeeping reasons. Group by `seed` and `framework`.

`equivalence_check.py` is run **first and deliberately**, not as a pre-flight check:
in this experiment the divergence it measures is a result (§4). It issues no
pass/fail, because Experiment 1 wants agreement and Experiment 2 expects divergence
and no single threshold serves both.

`check_network.py` must still **pass** — weight initialisation is untouched, so a
failure there would mean something else broke, and §4.4 confirms it passed with the
same fingerprints Experiment 1 recorded.

**`prepare_data.py` must be re-run after any Colab reconnect.** Tonic builds its cache
lazily; skipping it makes the first framework of the session absorb the entire 60k
conversion into its own training time.

Nine runs total: 3 frameworks × 3 seeds, all on 2026-08-06 between 22:05 and 23:56 in
a single session (~12–13 min per run). Merged with
`python collect_results.py --from <folder>/ex2 --experiment ex2`, plotted with
`python make_plots.py --experiment ex2`.

Environment: Tesla T4 · CUDA 12.8 · driver 580.82.07 · torch 2.11.0+cu128 ·
Python 3.12.13 · snntorch 1.0.0 · spikingjelly 0.0.0.0.14 · norse 1.1.0 · tonic 1.6.0.
The §4 equivalence run used the same package versions on CPU torch 2.13.0 for the
earlier iterations; the figures reproduced here are from the 2026-08-06 21:44:35 run
on torch 2.11.0+cu128, matching the training environment.

Outputs, all in this folder: [results/runs.csv](results/runs.csv),
[results/epochs.csv](results/epochs.csv), [results/layers.csv](results/layers.csv),
[results/runs/](results/runs/) (a full configuration snapshot per run),
[equivalence/](equivalence/) and [figures/](figures/).

The full pre-registered design, including every source read and the parameter
derivations behind §3, is [ex2_design.md](ex2_design.md).

---

## References

1. Reddi et al., *MLPerf Inference Benchmark*, arXiv:1911.02549. Throughput = queries/s (*Offline*); latency = time per single query (*Single-Stream*, 90th percentile).
2. Cheng, Hu, He & Huang (2025), *A comprehensive multimodal benchmark of neuromorphic training frameworks for spiking neural networks*, Engineering Applications of Artificial Intelligence. doi:10.1016/j.engappai.2025.111543
3. Pedersen et al. (2024), *Neuromorphic intermediate representation: A unified instruction set for interoperable brain-inspired computing*, Nature Communications 15:8122. doi:10.1038/s41467-024-52259-9
4. *High-performance deep spiking neural networks with 0.3 spikes per neuron*, Nature Communications (2024). doi:10.1038/s41467-024-51110-5
5. *Part-time Power Measurements: nvidia-smi's Lack of Attention*, arXiv:2312.02741. Sensor refresh 9.75–14.5 Hz; error up to 73 % against an external meter.
6. N-MNIST accuracy benchmarks: 98.74–98.93 % (earlier SNNs), 99.28–99.40 % (recent), 99.23 % (non-spiking CNN baseline).
7. Fang et al. (2023), *SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence*, Science Advances. doi:10.1126/sciadv.adi1480
8. Zenke & Ganguli (2018), *SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks*, Neural Computation 30, 1514–1541. doi:10.1162/neco_a_01086
9. Fang W. et al. (2021), *Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks*, ICCV 2021, pp. 2661–2671. Uses `init_tau = 2.0` on DVS128 Gesture — corroborates SpikingJelly's τ = 2.0 default for event data, from its own authors.
10. Eshraghian J.K. et al. (2023), *Training Spiking Neural Networks Using Lessons From Deep Learning*, Proceedings of the IEEE 111(9):1016–1054. The snnTorch paper. arXiv:2109.12894.

**Prior work in this project:** [experiments/ex1/report_ex1.md](../ex1/report_ex1.md)
(the forced-equivalence baseline and the ~0.15 pp noise floor used throughout),
[ex2_design.md](ex2_design.md) (the pre-registered design, hypotheses and sources),
`local_docs/norse_superspike_alpha_finding.md` (the SuperSpike `alpha` bug write-up),
`local_docs/controlled_variables.md` (the full held-constant table).
