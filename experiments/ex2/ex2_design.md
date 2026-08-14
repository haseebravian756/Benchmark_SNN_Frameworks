# Experiment 2 — Design: each framework out of the box

**Status: designed and ready to run. No training has happened.**
`config/config_ex2.yaml` is written and validated; the evidence figure in §2.3 is
already generated. Nothing else exists yet.

> ⚠️ **FRAMEWORK BUG affecting this experiment's Norse arm.** Norse 1.1.0's SuperSpike
> surrogate **accepts `alpha` and never reads it** — α = 1, 10 and 100 give
> byte-identical gradients, and it always behaves as α = 1. Fixed upstream (commit
> `1d2671a`, Aug 2024) but **never released**.
>
> ex2 deliberately does **not** work around this, because a user following Norse's
> defaults gets the buggy surrogate — so it is the honest out-of-the-box condition and
> becomes a *measured result* rather than a footnote. But it is a confound in any
> accuracy comparison: measured in ex1, the effective α = 1 gave Norse a conv1
> gradient norm **6.03×** the others', i.e. at a shared learning rate Norse trains at
> roughly 6× the step size. Discussed in §2.3; full write-up in
> `local_docs/norse_superspike_alpha_finding.md`.
>
> This is **one of two** framework bugs this project found; both are summarised at the
> top of `local_docs/SNNs_Introduction_BaseConcepts.md`. The other one does not affect
> ex2 (it is on SpikingJelly's cupy path, which ex2 does not use — see ex4).

Experiment 1 forced all three frameworks to compute *one* neuron, so that only the
implementation could differ. Experiment 2 asks the opposite question:

> **If you follow each framework's own defaults and its own documented example for
> event-camera data, what neuron do you actually get — and what does that do to
> accuracy, speed, spiking activity and memory?**

Dataset stays **N-MNIST**. Architecture, data pipeline and training recipe stay
exactly as Experiment 1. Only the **neuron parameters** move to each framework's own
out-of-the-box values. §5 explains why the scope is drawn there.

Every number below is either read from the installed source of the version we use,
or taken from official documentation, with a link. §9 lists every source once.

---

## 1. The neuron, in plain equations

### 1.1 What a leaky integrate-and-fire neuron does

Each timestep, a neuron holds one number — the **membrane potential** `v`. Three
things happen, in order:

| step | in words | in symbols |
|---|---|---|
| **leak** | forget part of what you were holding | `v ← d · v` |
| **integrate** | add the incoming signal | `v ← v + g · x` |
| **fire & reset** | if over threshold, emit a spike and drop back down | `if v > θ: spike, and reset v` |

Two numbers control the sub-threshold behaviour:

- **`d` — the decay** (per timestep). `d = 0` forgets everything instantly;
  `d = 1` never forgets. Always between 0 and 1.
- **`g` — the input gain.** How much of the incoming signal actually lands.

So the whole sub-threshold neuron is one line:

```
v[t] = d · v[t-1]  +  g · x[t]
```

### 1.2 Three numbers you can read off `d` and `g`

These let frameworks with different parameter names be compared directly. All three
are derived from `d` and `g` alone — no framework-specific knowledge needed.

**Effective time constant τ, in timesteps.** How long the neuron remembers. With
`v ← d·v` and nothing coming in, after `n` steps only `dⁿ` is left. Setting that
equal to the continuous form `exp(−n/τ)` gives:

```
τ = −1 / ln(d)                 [timesteps]
```

`d = 0.5` → τ ≈ 1.44 steps. `d = 0.9` → τ ≈ 9.49 steps. **A neuron with d = 0.9
remembers about 6.6× longer than one with d = 0.5.**

This matches snnTorch's own definition, β = e^(−Δt/τ), rearranged
([snnTorch Tutorial 3](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html)).

**DC gain — how excitable the neuron is.** Hold the input at a constant `I`. The
membrane climbs to a steady value and stops:

```
v* = g · I / (1 − d)           so   DC gain = g / (1 − d)
```

This is the number that decides **whether the neuron fires at all** for a given
input, and it is the one most easily missed, because it depends on `d` and `g`
*together*. A framework can have a perfectly ordinary-looking decay and still be
twice as trigger-happy as another.

**Whether decay and gain are independent.** Some frameworks let you set `g` freely;
one does not. See §4.

### 1.3 Reset: two kinds

| kind | what it does | after firing at `v = 1.05`, θ = 1.0 |
|---|---|---|
| **hard** (reset-to-value) | jump to a fixed value, normally 0 | `v = 0` |
| **soft** (reset-by-subtraction) | subtract the threshold, keep the remainder | `v = 0.05` |

Soft reset preserves "how far over threshold you were", so a strongly driven neuron
fires again sooner. They are genuinely different neurons, not a detail.

There is also a **timing** question — is the reset applied on the same timestep as
the spike, or the next one? snnTorch's default defers it (§2.1).

---

## 2. What each framework gives you out of the box

Three separate things must be distinguished, and mixing them up is the main way
this experiment could go wrong:

| label | meaning |
|---|---|
| **(D)** | **constructor default** — what you get by writing `Neuron()` with no arguments |
| **(E)** | **framework's own event-data example** — what its documented example for event-camera data passes explicitly |
| **(—)** | **no framework guidance exists**; we must choose and say so |

### 2.1 snnTorch 1.0.0 — `snn.Leaky`

Official equation, reset-by-subtraction (the default), from
[snn.Leaky docs](https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html):

```
U[t+1] = β·U[t] + I_in[t+1] − R·U_thr
```

So `d = β`, and **`g = 1` — the input enters at full strength, decoupled from the
decay.** Tutorial 3 states this explicitly: *"the effects of W and β are decoupled.
W is a learnable parameter that is updated independently of β."*

| parameter | out-of-box | kind | note |
|---|---|---|---|
| `beta` | **0.5** | (E) | **`beta` has NO constructor default** — snnTorch requires it. 0.5 is what its own N-MNIST tutorial uses ([Tutorial 7](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html)). |
| `threshold` | 1.0 | (D) | |
| `reset_mechanism` | `"subtract"` | (D) | **soft** reset |
| `reset_delay` | `True` | (D) | reset applied on the **following** timestep |
| `spike_grad` | `atan(alpha=2.0)` | (D) + (E) | `None` resolves to `atan()`; Tutorial 7 also passes `surrogate.atan()` |
| `learn_beta` | `False` | (D) | |
| `learn_threshold` | `False` | (D) | |

`beta` having no default is a real finding, not an inconvenience: **snnTorch has no
out-of-the-box time constant at all.** The honest substitute is the value its own
event-data tutorial uses, and that choice must be labelled as such in the report.

### 2.2 SpikingJelly 0.0.0.0.14 — `LIFNode`

Official equations, both branches, verbatim from the `LIFNode.__init__` docstring
(the same text as the [online neuron API docs](https://spikingjelly.readthedocs.io/zh-cn/latest/sub_module/spikingjelly.activation_based.neuron.html)):

```
decay_input == True :   H[t] = V[t-1] + (1/τ)·( X[t] − (V[t-1] − V_reset) )
decay_input == False:   H[t] = V[t-1] − (1/τ)·( V[t-1] − V_reset ) + X[t]
```

With `V_reset = 0`, both give `d = 1 − 1/τ`. They differ only in the gain:
**`decay_input=True` → `g = 1/τ`; `decay_input=False` → `g = 1`.**

| parameter | out-of-box | kind | note |
|---|---|---|---|
| `tau` | **2.0** | (D) + (E) | its DVS-Gesture example does not pass `tau`, so the default stands. Independently, Fang et al. 2021 (ICCV) use `init_tau 2.0` on DVS128 Gesture. |
| `decay_input` | **True** | (D) + (E) | not passed by the example → default. Makes `g = 0.5`. |
| `v_threshold` | 1.0 | (D) | |
| `v_reset` | 0.0 | (D) | **hard** reset (`None` would mean soft) |
| `surrogate_function` | **`ATan(alpha=2.0)`** | (E) | ⚠ the *constructor* default is `Sigmoid(alpha=4.0)`, but the framework's own event example passes `ATan()`. **These disagree.** |
| `detach_reset` | **True** | (E) | ⚠ constructor default is `False`; the event example passes `True`. **These disagree.** |
| `step_mode` | `'s'` | forced | our shared network feeds one timestep at a time. See §6. |

The two ⚠ rows are the most interesting thing in this table: **SpikingJelly's own
recommended practice for event data differs from its own constructor defaults.**
Which to use is a design decision, taken in §3.

### 2.3 Norse 1.1.0 — `LIFBoxCell`

Official ODE, from the
[LIFBoxCell docs](https://norse.github.io/norse/generated/norse.torch.module.lif_box.LIFBoxCell.html)
and the installed source:

```
v̇ = (1/τ_mem)·( v_leak − v + i )
```

One Euler step with timestep `dt`:

```
v[t] = v[t-1] + dt·τ_mem_inv·( x[t] + v_leak − v[t-1] )
     = (1 − dt·τ_mem_inv)·v[t-1]  +  (dt·τ_mem_inv)·x[t]
```

So with `a = dt · τ_mem_inv`:  **`d = 1 − a` and `g = a`.**

**`d` and `g` are locked: `g = 1 − d`.** They cannot be set independently, because
both come from the single product `dt · τ_mem_inv`. This is a structural property of
the discretised ODE, not a missing feature — and it is why Experiment 1 needed an
extra `input_scale` factor to reach gain 1.0. Out of the box there is no such
factor.

| parameter | out-of-box | kind | note |
|---|---|---|---|
| `dt` | 0.001 | (D) | |
| `tau_mem_inv` | 100.0 | (D) | so `a = 0.1` → `d = 0.9`, `g = 0.1` |
| `v_th` | 1.0 | (D) | |
| `v_reset` | 0.0 | (D) | |
| `v_leak` | 0.0 | (D) | |
| `reset_method` | `reset_value` | (D) | **hard** reset |
| `method` | `'super'` (SuperSpike) | (D) | see the warning below |
| `alpha` | 100.0 | (D) | **has no effect — see below** |

**Two things must be stated plainly.**

**(a) Norse ships no event-camera example.** Its bundled tasks are `cartpole`,
`cifar10`, `correlation_experiment`, `memory`, `mnist`, `mnist_pl` and
`speech_commands` — verified by listing `norse/task/` in the installed package.
So for Norse, kind **(E)** does not exist. Its out-of-the-box position is its
constructor defaults and nothing else, and the report must say so rather than
inventing guidance.

**(b) The default surrogate's `alpha` is silently ignored.** `method='super'` with
`alpha=100.0` does *not* behave like a sharp surrogate. Re-verified in 1.1.0: the
backward pass is `grad_output / (|x| + 1)²` with `alpha` never read, so
`alpha = 1, 10, 100` produce byte-identical gradients — it always behaves as
`alpha = 1`. Fixed upstream in commit `1d2671a` (Aug 2024), **never released**;
v1.1.0 remains current on PyPI. Full write-up:
`local_docs/norse_superspike_alpha_finding.md`.

Experiment 1 deliberately avoided this by using `circ(0.5)`. **Experiment 2 must
not avoid it** — a user following the defaults gets the buggy surrogate, so that is
the honest out-of-the-box condition. The bug stops being a footnote and becomes a
measured result.

**Evidence figure — already generated:**

```powershell
.venv\Scripts\python probe_norse_alpha.py --experiment ex2
```

![Norse SuperSpike ignores its alpha](figures/FX.1_norse_superspike_alpha.png)

`probe_norse_alpha.py` is a **standalone, single-purpose module.** It reads no CSV,
touches no config, and nothing in the training pipeline imports it — deleting it
cannot break anything. It exists separately because the bug is a property of the
library, not of any run, so it does not belong in the results pipeline. **Safe to
delete once Norse ships a release containing commit `1d2671a`**; until then the
figure belongs in the final report next to Norse's numbers.

Every curve in panels A and C is **measured** — a tensor is pushed through the real
surrogate and `.backward()` is called — so no hand-copied formula can make the figure
disagree with the installed code. Panel B is the published form, which is the one
place a formula is needed, because the shipped code cannot produce it.

What the three panels show:

| panel | shows |
|---|---|
| **A** | the shipped surrogate at α = 1, 10, 100. Three curves are drawn; you see one. Largest difference between any two: **0.00e+00** — not small, exactly zero. |
| **B** | the same three α through the published formula. Largest difference: **0.88**. α is meant to matter across almost the whole range. |
| **C** | the shipped default against `atan(2.0)` (what the other two frameworks use) and `circ(0.5)` (what ex1 used instead). |

Panel C carries the number that matters. All three surrogates agree *at* threshold
and diverge in the **tails**:

| \|v − θ\| | super, shipped | atan(2.0) | circ(0.5) | super ÷ atan |
|---|---|---|---|---|
| 0.0 | 1.0000 | 1.0000 | 1.0000 | 1.00× |
| 0.5 | 0.4444 | 0.2884 | 0.3536 | 1.54× |
| 1.0 | 0.2500 | 0.0920 | 0.0894 | **2.72×** |
| 2.0 | 0.1111 | 0.0247 | 0.0143 | **4.50×** |

Most neurons sit far from threshold most of the time, which is why this tail
difference compounds into ex1's measured **6.03×** gradient-norm ratio on the real
network's conv1. It is also why `circ(0.5)` was chosen for ex1: at |x| = 1.0 it gives
0.0894 against atan's 0.0920 — a much closer match than the default's 0.2500.

**A note on what is *not* claimed here.** The figure deliberately does not report a
gradient norm through a synthetic layer. A made-up layer's weight scale decides how
far its pre-activations sit from threshold, and since the surrogates differ only in
their tails, such a number would describe the arbitrary layer more than the
surrogates. The 6.03× figure comes from ex1's measurement on the actual network over
8 batches, and that is the only real-effect number quoted.

### 2.4 sinabs 3.1.3 — and the answer is not a number

**Added 2026-08-13.** sinabs was researched for ex1 first; this section records what
"out of the box" turns out to mean for it, which is qualitatively different from the
other three.

**`sinabs.layers.LIF` has no default time constant.** `tau_mem` is a required
positional argument — exactly snnTorch's `beta` situation — so **kind (D) does not
exist** for it.

**Kind (E) does not exist either, and that is the finding.** Every one of sinabs' own
examples uses **`IAF`, never `LIF`**. All four were checked:

| sinabs' own example | dataset | neuron constructed |
|---|---|---|
| `docs/tutorials/nmnist.ipynb` | **N-MNIST** (BPTT via EXODUS) | `backend.IAFSqueeze(batch_size=batch_size, min_v_mem=-1)` |
| `docs/speck/notebooks/nmnist_quick_start.ipynb` | **N-MNIST** | `sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential())` |
| `examples/visualizer/gesture_viz.py` | **DVS128 Gesture** | ANN + `from_model()` → IAF |
| `docs/tutorials/bptt.ipynb` | Sequential MNIST | ANN + `from_model()` → IAF |

And `from_model()` — sinabs' headline ANN→SNN converter — defaults to
`spike_layer_class = sl.IAFSqueeze` and `min_v_mem = -1.0`
(`sinabs/from_torch.py:20,26`, read from the installed package).

**So there is no out-of-the-box leaky sinabs neuron to compare.** Out of the box,
sinabs gives you an **integrator**.

**IAF is literally LIF with the leak switched off**, which is what makes this
expressible in the shared pipeline without a second adapter. `sinabs/layers/iaf.py`
calls `super().__init__(tau_mem=np.inf, ..., norm_input=False)`, so
α = e^(−1/∞) = 1. Verified on the installed package: `IAF()` reports
`alpha_mem_calculated = 1.0`. The ex2 config therefore writes `tau_mem: .inf`, which
reproduces sinabs' own IAF exactly through the `LIF` class the adapter already uses.

`norm_input: false` is not a preference here but **mathematically forced**: with
α = 1 the normalised gain would be 1 − α = **0** and the neuron would be completely
deaf. sinabs' IAF hardcodes it for the same reason.

The out-of-the-box values, with their labels:

| setting | value | kind | source |
|---|---|---|---|
| `tau_mem` | `.inf` | **(E)** | all four examples above; also (D) of `from_model`'s `spike_layer_class` |
| `norm_input` | `false` | forced | 1 − α = 0 would make the neuron deaf; IAF hardcodes it |
| `spike_threshold` | `1.0` | (D) | `LIF` and `from_model` |
| `spike_fn` | `MultiSpike` | (D) | several spikes per timestep permitted |
| `reset_fn` | `MembraneSubtract()` | (D) | **soft** reset, remainder kept |
| `min_v_mem` | `-1.0` | **(E)** + (D) | both N-MNIST tutorials pass it; also `from_model`'s default |
| `surrogate` | `PeriodicExponential()` | **(E)** | the N-MNIST BPTT example; (D) would be `SingleExponential` |
| `tau_syn` | `None` | (D) | first-order |

Two notes on those choices. **`PeriodicExponential` over the (D) `SingleExponential`**
follows the same rule already applied to SpikingJelly's ATan in §2.2 — prefer (E)
where the framework's own event-data example is explicit — and it is the coherent
partner to `MultiSpike`, because it repeats its gradient window at every multiple of
the threshold, where a single-window surrogate gives no gradient at all to a
timestep's 2nd or 3rd spike. **`min_v_mem = -1.0` is an asymmetry worth stating:**
none of the other three frameworks offers a membrane floor at all.

---

## 3. The four neurons on one scale

Converting §2 through §1.2. **Measured, not calculated on paper** — each neuron was
built with its out-of-the-box arguments, its threshold raised out of reach to
isolate the linear filter, and driven with a constant input:

| | snnTorch | SpikingJelly | Norse | sinabs |
|---|---|---|---|---|
| written as | `beta=0.5` | `tau=2.0`, `decay_input=True` | `dt=0.001`, `tau_mem_inv=100` | `tau_mem=inf` (= `IAF`) |
| **decay `d`** | 0.5000 | 0.5000 | 0.9000 | **1.0000** |
| **input gain `g`** | 1.0000 | 0.5000 | 0.1000 | 1.0000 |
| **DC gain `g/(1−d)`** | **2.0000** | 1.0000 | 0.9998 | **undefined** (1/0) |
| **τ (timesteps)** | 1.44 | 1.44 | 9.49 | **∞** |
| reset | **soft**, delayed 1 step | hard | hard | **soft** |
| spikes per step | 1 | 1 | 1 | **unbounded** |
| membrane floor | none | none | none | **−1.0** |
| surrogate | atan(2.0) | ATan(2.0) *(example)* / Sigmoid(4.0) *(default)* | super, effectively α=1 | PeriodicExponential *(example)* |

Four things fall out, and they are the experiment's starting hypotheses:

**0. sinabs is not a leaky neuron at all.** Its DC gain cannot be computed — `1/(1−1)`
— because the membrane never forgets, so a constant input accumulates without bound
instead of settling. That column is genuinely empty for sinabs rather than large.
Measured consequence on the `constant_step` input (amplitude 0.15, threshold 1.0):
snnTorch, SpikingJelly and Norse plateau at 0.30, 0.15 and 0.15 and **never fire**,
while sinabs accumulates past 1.0 at step 16 and fires **13 times**. The comparison is
"three leaky neurons and one integrator", by the libraries' own defaults.

This is not sinabs being careless. No-leak **+** multi-spike **+** subtract-reset are
exactly the three conditions under which a spiking neuron's firing rate equals ReLU —
the ANN-conversion guarantee sinabs exists to provide. Its defaults are a *conversion*
choice, not a neuron-modelling one. Derivation:
`local_docs/SNNs_Introduction_BaseConcepts.md` §4.

⚠️ **`MultiSpike` breaks the spike-rate column for sinabs.**
`src/adapters/base.py` documents `spike_rate()` as a fraction of spikes per neuron per
timestep; with several spikes allowed per timestep it can exceed 1.0. **Report
sinabs' spike rate, do not rank it against the other three.**

**1. Two of the three agree on decay, by coincidence.** snnTorch's tutorial β = 0.5
and SpikingJelly's default τ = 2.0 give **identical** decay and identical τ. Nobody
coordinated this; `1 − 1/2 = 0.5` simply lands on the same number.

**2. But snnTorch is twice as excitable.** Its DC gain is 2.0 against 1.0 for the
other two, because its gain is decoupled from its decay while theirs are normalised.
Concretely, driven with a constant 0.6 against a threshold of 1.0:

| | steady-state `v` | fires? |
|---|---|---|
| snnTorch | 0.6 × 2.0 = **1.2** | **yes**, repeatedly |
| SpikingJelly | 0.6 × 1.0 = 0.6 | no |
| Norse | 0.6 × 1.0 = 0.6 | no |

Verified by simulation: at input 0.6, snnTorch fires on steps 3 and 7 while the
other two never reach threshold. **Expect snnTorch's spike rate to be markedly
higher than the other two.**

**3. Norse remembers ~6.6× longer.** τ = 9.49 steps against 1.44. Over T = 20
timesteps, Norse integrates across roughly half the sample while the other two see a
sliding window of two or three frames.

---

## 4. Where the frameworks structurally cannot be made to agree

Not parameter choices — properties of the maths. Worth stating because they bound
what any "fair" comparison can mean.

| | constraint |
|---|---|
| **Norse** | `g = 1 − d`. Decay and gain are one parameter. Choosing decay 0.9 forces gain 0.1. |
| **SpikingJelly** | `g ∈ {1/τ, 1}` only, via the `decay_input` flag. No free choice in between. |
| **snnTorch** | `g = 1` always; `d = β` free. The only one of the three with a fully decoupled gain. |

So the three frameworks do not span the same parameter space. Experiment 1 reached a
common point (`d = 0.9`, `g = 1`) only by adding an external `input_scale = 10.0` to
Norse — a factor outside the neuron. Experiment 2 removes that scaffolding, which is
the point.

---

## 5. Scope: what changes, and what deliberately does not

### 5.1 The decision

**Only the neuron block changes.** Data pipeline, architecture, optimiser, learning
rate, loss, timesteps, batch size and seeds stay byte-identical to Experiment 1.

### 5.2 Why not "each framework's whole recipe"

Because the answer would be uninterpretable. Each framework's example uses a
different recipe entirely — verified by reading them:

| | snnTorch Tutorial 7 | SpikingJelly `classify_dvsg.py` |
|---|---|---|
| dataset | N-MNIST | DVS128 Gesture |
| timesteps | `ToFrame(time_window=1000)` | `T = 16` |
| batch size | 128 | 16 |
| optimiser | Adam, lr 2e-2 | Adam or SGD, lr 0.1 |
| loss | `mse_count_loss(0.8, 0.2)` | voting layer |
| architecture | 12C5-MP2-32C5-MP2-FC10 | 5×(128C3-BN-MaxPool), 2 FC, dropout 0.5 |
| epochs | 1 | 64 |

Adopt all of that per framework and every metric becomes incomparable: different
batch sizes change wall-clock time and memory, different architectures change
parameter counts, different losses change accuracy. You would be comparing training
recipes, not frameworks — and no metric in `runs.csv` would mean the same thing in
two rows.

Holding everything except the neuron fixed keeps every metric comparable **and**
isolates the question actually being asked: *what does the framework's default
neuron do to my results?*

Worth noting the architecture question is already settled: **Experiment 1's
architecture came from snnTorch Tutorial 7** — `Conv2d(2,12,5)` → `MaxPool2d(2)` →
neuron → `Conv2d(12,32,5)` → `MaxPool2d(2)` → neuron → `Flatten` →
`Linear(32·5·5, 10)` → neuron. Now verified directly against the tutorial source,
having previously been an unverified claim.

### 5.3 The one judgement call, stated openly

**SpikingJelly's constructor defaults and its own event-data example disagree**
(surrogate, and `detach_reset`). Only one can be "out of the box".

**Recommendation: follow the example, not the bare constructor** — `ATan(2.0)` and
`detach_reset=True`. Reason: the experiment's question is "what happens if you follow
what the framework tells you to do for event data", and a documented, shipped example
*is* what it tells you. A bare `LIFNode()` is what you get if you read nothing.

This is a judgement call, not a fact, and it is cheap to check both ways — so
**run both**, as ex2 and ex2b. The difference is exactly the measured cost of
"read the docs" versus "accept the defaults", which is itself a result worth having.

### 5.4 Learning rate — a known risk to flag now

Experiment 1 used one shared learning rate (2e-3) and recorded as limitation L7 that
it may suit one framework's gradient scale better than another's. Experiment 2 makes
that risk **larger**, because the surrogate gradients now genuinely differ (atan(2)
vs Sigmoid(4)-or-ATan(2) vs an effectively-α=1 SuperSpike). Keep the shared learning
rate — changing it per framework would reintroduce the recipe problem from §5.2 —
but expect to state plainly that a framework may be underperforming because of a
gradient-scale mismatch rather than anything intrinsic. The measurement that would
settle it is a per-framework gradient-norm probe, which `equivalence_check.py`
already does.

---

## 6. What stays forced, and why

Two carry-overs from Experiment 1 are **not** out-of-the-box, and both must be
declared:

| forced | why it cannot change |
|---|---|
| **Norse uses `LIFBoxCell`, not `LIFCell`** | Norse's headline LIF is `LIFCell`, a **second-order** neuron with an extra synaptic-current state. `LIFBoxCell` is the first-order one that structurally matches the other two. Switching to `LIFCell` would change the neuron's order, not just its parameters — a different experiment. Carried over from ex1 for comparability, and **it means Norse is not strictly "out of the box" even here.** State this as a limitation. |
| **SpikingJelly runs `step_mode='s'`, `backend='torch'`** | The shared network loops over timesteps and hands each layer one frame. SpikingJelly's fused CUDA kernel (`cupy`) needs `'m'` and the whole `[T,…]` tensor. Verified in ex1: `'m'` and looped `'s'` produce identical spikes, so this is a speed-only limitation, not a correctness one. Same limitation as ex1 (L2). |

---

## 7. What to expect — hypotheses, not conclusions

Written down in advance so the results can contradict them.

| # | expectation | reasoning |
|---|---|---|
| H1 | **The neurons diverge, and that divergence is the headline result.** | The three now have different decay, gain and reset. **Confirmed:** `equivalence_check.py` measures worst max\|dv\| ≈ **2.3e-01** against ex1's 1.2e-07, first divergence at the timestep the input arrives. The script issues no pass/fail — it reports the numbers and plots, because ex1 wants agreement and ex2 expects divergence, so no single threshold serves both. Report the measurement, do not "fix" it. |
| H2 | **snnTorch's spike rate is much higher than the other two.** | DC gain 2.0 vs 1.0 (§3). Directly measurable, and the clearest single prediction here. |
| H3 | **Norse's activity is smoothest over time; snnTorch's is burstiest.** | τ = 9.49 vs 1.44 steps. Per-timestep spike rates would show this directly — currently not recorded (metrics item M3). |
| H4 | **Accuracy differences are larger than Experiment 1's ~0.15 pp noise floor.** | In ex1 the frameworks computed the same function, so accuracy could only differ by noise. Here they compute genuinely different neurons, so a real difference is possible for the first time. **The ex1 noise floor is what makes this claim testable** — reuse it, do not re-derive it. |
| H5 | **Speed, memory and latency rankings are roughly unchanged from ex1.** | These are properties of each library's implementation, and the parameter values do not change the amount of arithmetic done. If a ranking *does* move, that is interesting and needs explaining. |
| H6 | **Norse is handicapped by the SuperSpike bug, not by Norse.** | Effective α = 1 has much fatter gradient tails than atan(2) — ex1 measured a 6.03× gradient-norm ratio. Any Norse accuracy shortfall must be attributed carefully, and the fix (`circ(0.5)`, i.e. ex1's setting) is the control that separates the two explanations. |

**H4 and H6 together are the reason this experiment is worth running:** ex1 proved
the frameworks are equivalent when forced. If accuracy now diverges, ex2 shows how
much of that divergence a user inherits by default — and H6 says part of it is a
released bug rather than a design choice.

---

## 8. Concrete plan

### 8.1 Config

`config/config_ex2.yaml` — **written and validated**. It extends `config/default.yaml`
and overrides only the `neuron:` block:

```yaml
neuron:
  snntorch:
    beta: 0.5                 # (E) Tutorial 7 — snnTorch has NO default beta
    threshold: 1.0            # (D)
    reset_mechanism: subtract # (D)  soft  <- differs from ex1
    reset_delay: true         # (D)  delayed <- differs from ex1
    surrogate: { type: atan, alpha: 2.0 }   # (D)+(E)

  spikingjelly:
    tau: 2.0                  # (D)+(E)  <- differs from ex1 (was 10.0)
    decay_input: true         # (D)+(E)  gain 0.5  <- differs from ex1
    v_threshold: 1.0          # (D)
    v_reset: 0.0              # (D)  hard
    detach_reset: true        # (E)  <- differs from ex1 and from the (D) False
    step_mode: s              # forced, see §6
    backend: torch            # forced, see §6
    surrogate: { type: atan, alpha: 2.0 }   # (E)

  norse:
    dt: 0.001                 # (D)
    tau_mem_inv: 100.0        # (D)  -> decay 0.9, gain 0.1
    v_th: 1.0                 # (D)
    v_reset: 0.0              # (D)
    v_leak: 0.0               # (D)
    reset_method: value       # (D)  hard
    input_scale: 1.0          # NO rescaling  <- differs from ex1 (was 10.0)
    surrogate: { type: super, alpha: 100.0 } # (D) — alpha is ignored, see §2.3
```

**No code changes are needed.** Verified by building all three adapters from exactly
the block above: `extends:` is already supported by `src/config.py`, `super` is
already in Norse's surrogate registry, `sigmoid` is already in SpikingJelly's, and
`input_scale: 1.0` passes through as a genuine no-op. All three construct cleanly and
report:

```
snnTorch      beta 0.5,  threshold 1.0, subtract, reset_delay True, atan(2.0)
SpikingJelly  tau 2.0,   decay_input True, v_reset 0.0, detach_reset True, atan(2.0)
Norse         dt 0.001,  tau_mem_inv 100.0, value reset, super(100.0)
              effective_decay_gain 0.900/0.100
```

That last line is worth noting: the adapter already *prints* the effective decay and
gain, so the §3 divergence is visible in the run's own config snapshot without any
extra work.

A second config `config/config_ex2_bare.yaml` for §5.3's ex2b differs in exactly
two lines: SpikingJelly's `surrogate: {type: sigmoid, alpha: 4.0}` and
`detach_reset: false`.

### 8.2 Run order

```powershell
# 0. Evidence figure for Norse's default surrogate — local, no GPU, already done
.venv\Scripts\python probe_norse_alpha.py --experiment ex2

# 1. Record the divergence FIRST — this is the result, not a pre-flight check
#    (no pass/fail is issued; it prints the deviations and writes the plots)
python equivalence_check.py --config config/config_ex2.yaml --experiment ex2

# 2. Confirm the networks still start from identical weights (must still PASS —
#    weight init is untouched, so a failure here means something else broke)
python check_network.py --all --config config/config_ex2.yaml

# 3. Train: 3 frameworks x 3 seeds, same as ex1
python prepare_data.py --config config/config_ex2.yaml   # ALWAYS, after any reconnect
python train.py --config config/config_ex2.yaml --experiment ex2 \
    --framework <fw> --seed <s> --results-root <drive>

# 4. Home, then plots
python collect_results.py --from <folder>/ex2 --experiment ex2
python make_plots.py --experiment ex2
```

**Three seeds again.** Experiment 1 showed one seed publishes false rankings, and
that lesson does not become invalid because the neuron changed.

### 8.3 What the existing tooling gives for free

- `make_plots.py --experiment ex2` works unchanged — the plotting schema's condition
  axis is already generic.
- **F6.1 (effect size vs noise) is the figure that decides H4.** It expresses each
  difference in units of that metric's own run-to-run spread, which is exactly the
  test "is this bigger than ex1's noise floor?"
- **F2.2 (paired differences)** decides whether any new ranking is real or a swap.
- The equivalence figures will now show three *separated* traces instead of one
  overlapping line — the same plot carrying the opposite message. Worth putting
  side by side with ex1's in the final report.

### 8.4 Deliverables

`experiments/ex2/` with `report_ex2.md`, `equivalence/`, `results/`, `figures/` —
same layout as ex1, same report structure.

### 8.5 After ex2 — two decisions already queued

**Promote these values into `default.yaml`?** If ex2's picture turns out to be the
more authentic one — these being the values the frameworks themselves put forward —
the plan is to make them the project baseline and keep ex1's forced-equivalence
config as the special case rather than the default. Deferred until the results exist;
noted in `config/config_ex2.yaml` so the intent is not lost.

**SpikingJelly in its fastest mode** (open item E5). Every run so far has measured
SpikingJelly in `step_mode='s'` with the `torch` backend, which cannot reach its fused
CUDA kernel — so **its measured speed advantage is a lower bound**, and the 11×
training acceleration its own paper claims has never been exercised here. The
follow-up runs `step_mode='m'` with the `cupy` backend. It is not a config flag: the
network forward must hand the whole `[T, …]` tensor to each layer, with the conv and
pooling layers wrapped (`layer.*` / `SeqToANNContainer`). Verified in ex1 that `'m'`
and looped `'s'` produce identical spikes, so **accuracy must not move** — if it
does, the port is wrong. That makes it a clean speed-only experiment with a built-in
correctness check.

---

## 9. Sources

Everything above traces to one of these. Marked **[installed]** where the claim was
verified by reading the installed package (authoritative for the exact versions we
run) rather than a web page.

### Framework documentation

| # | source |
|---|---|
| S1 | **snnTorch `snn.Leaky` API** — parameter list, defaults, both reset equations, `spike_grad=None` → ATan. https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html · also **[installed]** snntorch 1.0.0 |
| S2 | **snnTorch Tutorial 3** — β = e^(−Δt/τ); `U[t+1] = βU[t] + WX[t+1]`; the explicit statement that W and β are decoupled. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html |
| S3 | **snnTorch Tutorial 7, Neuromorphic Datasets with Tonic** — the N-MNIST example: `beta = 0.5`, `surrogate.atan()`, `Denoise(filter_time=10000)`, `batch_size = 128`, Adam lr 2e-2, `mse_count_loss(0.8, 0.2)`, and the 12C5-MP2-32C5-MP2-FC10 architecture. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html |
| S4 | **SpikingJelly `LIFNode` API** — both `decay_input` equations verbatim, all defaults, `v_reset=None` meaning soft reset. https://spikingjelly.readthedocs.io/zh-cn/latest/sub_module/spikingjelly.activation_based.neuron.html · also **[installed]** spikingjelly 0.0.0.0.14 |
| S5 | **SpikingJelly `classify_dvsg.py`** — its own event-camera example: `LIFNode` with `surrogate.ATan()` and `detach_reset=True`, `tau`/`decay_input` left at defaults, T=16, batch 16, 128 channels, lr 0.1, 64 epochs. **[installed]** at `spikingjelly/activation_based/examples/classify_dvsg.py`; docs mirror at https://github.com/fangwei123456/spikingjelly |
| S6 | **Norse `LIFBoxCell` / `LIFBoxParameters` API** — the ODE, and defaults `tau_mem_inv=100`, `v_leak=0`, `v_th=1`, `v_reset=0`, `method='super'`, `alpha=100`, `reset_method=reset_value`, `dt=0.001`. https://norse.github.io/norse/generated/norse.torch.module.lif_box.LIFBoxCell.html and https://norse.github.io/norse/generated/norse.torch.module.lif_box.LIFBoxParameters.html · also **[installed]** norse 1.1.0 |
| S7 | **Norse, Parameter learning in SNNs** — treats these as hyperparameters with "arbitrary defaults", and warns that "optimisation works best if the scale of the values to be optimised is in an appropriate range. This is not the case here for the inverse time constant, which has a value of > 100." https://norse.github.io/norse/pages/parameters.html |
| S8 | **Norse bundled tasks** — `cartpole`, `cifar10`, `correlation_experiment`, `memory`, `mnist`, `mnist_pl`, `speech_commands`. **No event-camera task.** **[installed]** `norse/task/` |

### Publications

| # | source |
|---|---|
| S9 | Fang W., Yu Z., Chen Y., Masquelier T., Huang T., Tian Y. (2021), *Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks*, **ICCV 2021**, pp. 2661–2671. Uses `init_tau = 2.0` on DVS128 Gesture — independent corroboration of S4/S5's τ = 2.0 for event data, from SpikingJelly's own authors. https://openaccess.thecvf.com/content/ICCV2021/html/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.html |
| S10 | Eshraghian J.K. et al. (2023), *Training Spiking Neural Networks Using Lessons From Deep Learning*, **Proceedings of the IEEE** 111(9):1016–1054. The snnTorch paper. arXiv:2109.12894. *(Cited for provenance only — the PDF is image-based, so the β↔τ relation used here is taken from S2, which states it directly.)* |

### Our own prior measurements

| # | source |
|---|---|
| S11 | `experiments/ex1/report_ex1.md` — the ~0.15 pp accuracy noise floor (three agreeing estimates), the ex1 forced-neuron configuration, and limitations L2 and L7 which carry into ex2. |
| S12 | `local_docs/norse_superspike_alpha_finding.md` — the SuperSpike `alpha` bug: measured 6.03× gradient-norm ratio, byte-identical gradients for α = 1/10/100, upstream commit `1d2671a` unreleased. **Re-verified in norse 1.1.0 while writing this document.** |

### Measured while writing this document

The §3 table (decay, gain, DC gain, τ) and the §3 firing example were **measured**,
not taken from any document: each neuron was constructed with its out-of-the-box
arguments, its threshold raised beyond reach to isolate the linear filter, and driven
with a constant input. Every measured value matched the value predicted from the
official equations to 4 decimal places, which is itself a check that §1–§2 read the
documentation correctly.

---

## 10. Open questions for you

1. **§5.3 — run both variants?** Recommendation is yes: ex2 follows SpikingJelly's
   event-data example, ex2b its bare constructor defaults. Costs one extra ~45 min
   seed sweep and measures the value of reading the docs.
2. **snnTorch's `beta`.** 0.5 comes from Tutorial 7 (its own N-MNIST example), which
   is the best available evidence — but it is an (E) value standing in for a
   non-existent (D) one. Accept, or prefer some other justification?
3. **Norse `LIFBoxCell` vs `LIFCell`** (§6). Keeping `LIFBoxCell` preserves
   comparability with ex1 but means Norse is not strictly out-of-the-box. The
   alternative is a genuinely different (second-order) neuron. Recommendation: keep
   `LIFBoxCell`, declare the limitation.
