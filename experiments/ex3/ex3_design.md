# Experiment 3 — Design: each framework's specialised neuron on DVS128 Gesture

**Status: research and design only. Nothing has been built and nothing has been run.**
No config file exists, no code has been changed, no data has been downloaded. This
document is the argument for the experiment, not a report on it.

> ⚠️ **FRAMEWORK BUG relevant to this design.** Norse 1.1.0's SuperSpike surrogate
> **accepts `alpha` and never reads it** — α = 1, 10 and 100 give byte-identical
> gradients, and it always behaves as α = 1. Fixed upstream (commit `1d2671a`, Aug
> 2024) but **never released**. This is the reason §6 forces `circ(0.5)` here rather
> than using `super`; see T22 in the sources and
> `local_docs/norse_superspike_alpha_finding.md`.
>
> One of **two** framework bugs found in this project — both summarised at the top of
> `local_docs/SNNs_Introduction_BaseConcepts.md`. If ex3 ever runs SpikingJelly with
> `backend: cupy`, the second one applies too (see `experiments/ex4/ex4_design.md`).

Experiments 1 and 2 both asked about **one neuron** — first forced to agree (ex1),
then allowed to differ by each framework's own defaults (ex2). Both ran on N-MNIST.
Experiment 3 changes both halves of that at once, on purpose:

> **When the task genuinely requires memory across time, what specialised neuron does
> each framework offer for it — and what does choosing that neuron buy you, in
> accuracy, speed, spiking activity and memory?**

This merges open items **E3** (second dataset) and **E4** (other neuron types) from
`docs/open_items.md`. They are merged because separately neither is interesting:
a second dataset with the same neuron measures the dataset, and a fancier neuron on
N-MNIST measures nothing at all, because N-MNIST does not need one (§1.5).

Every claim below is either read from the installed source of the version we run, or
taken from official documentation, with a link. §9 lists every source once.

---

## 1. Where this experiment sits

### 1.1 Words used in this document

Short definitions, so nothing below depends on you already knowing the vocabulary.

| word | one line |
|---|---|
| **event camera / DVS** | A camera with no frames. Each pixel independently reports the moment its brightness changes. "DVS" = Dynamic Vision Sensor. |
| **event** | One report from one pixel: `(x, y, timestamp, polarity)`. |
| **polarity** | Whether the brightness went **up** (+1, "ON") or **down** (−1, "OFF"). This is why event data has 2 channels where a grey image has 1. |
| **sparsity** | The fraction of the data that is empty. An event camera looking at a still scene outputs almost nothing, so its data is *sparse* — mostly zeros. This is the property SNNs are supposed to exploit. |
| **framing / binning** | Turning a continuous stream of events back into a stack of pictures by adding up all events that fall in the same time slice. `ToFrame` in Tonic. Necessary because a `Conv2d` needs a rectangular tensor. |
| **T (timesteps)** | How many of those frames one sample becomes. The network is run T times per sample. |
| **saccade** | A small, fast eye movement. N-MNIST was made by moving the camera in three saccades across a printed digit, so that a *static* image produces events at all. |
| **ego-motion** | Motion of the camera itself, as opposed to motion in the scene. Ego-motion makes every edge in the scene generate events, which destroys sparsity. |
| **state variable** | A number a neuron carries from one timestep to the next. A plain LIF has one (membrane potential `v`). More state = longer/richer memory, and more memory used per neuron. |
| **first-order / second-order neuron** | First-order = one state variable. Second-order = two (typically membrane *and* synaptic current). |
| **time constant (τ)** | How long a neuron remembers, measured in timesteps. Big τ = long memory. Formally `τ = −1/ln(d)` where `d` is the per-step decay. |
| **adaptation** | The neuron gets *harder to fire* the more it has recently fired — either by raising its own threshold or by growing an inhibitory current. A built-in "calm down" mechanism. |
| **refractory period** | The short time after a spike during which a real neuron cannot fire again. Adaptation is a soft, graded version of this. |
| **surrogate gradient** | A spike is a step function, whose derivative is zero everywhere and infinite at one point — useless for backpropagation. A surrogate gradient substitutes a smooth, fake derivative so gradient descent can work. Affects the backward pass only. |
| **temporal credit assignment** | The problem of working out *which earlier moment* was responsible for the final answer. Gets exponentially harder the longer the sequence. |
| **noise floor** | How much a measurement moves purely from run-to-run randomness. A difference smaller than the noise floor is not a result. Experiment 1 measured ours at ~0.15 pp accuracy. |
| **effect size** | A difference expressed in units of that metric's own noise floor, so "is this real?" has a numeric answer. |
| **class-conditional** | Depending on which class the sample belongs to. Used below in "the temporal order is class-conditional" = the order of events is what distinguishes the classes. |

### 1.2 Which kind of task is this? A one-screen taxonomy

Neuromorphic-vision benchmarks fall into a few recognisable families. Knowing which
family a dataset belongs to tells you, before you write any code, whether time matters.

| category | what the model must output | does temporal order matter? | typical datasets |
|---|---|---|---|
| **A. Converted static classification** | one class label per sample | **No.** A static image was converted to events by moving the camera. Summing all events back into one picture recovers the original image. | **N-MNIST**, N-Caltech101, CIFAR10-DVS |
| **B. Native (real-world) object classification** | one class label per sample | Barely. Real objects recorded with a real camera, but the class is still an appearance, not a movement. | ASL-DVS, N-Cars, PokerDVS |
| **C. Action / gesture recognition** | one class label per sample | **Yes, decisively.** The label is a *movement*. Collapse time and the classes become indistinguishable. | **DVS128 Gesture**, DailyDVS-200, DVS-Lip |
| **D. Dense regression / prediction** | a value per pixel, per moment | Yes, continuously. | optical flow (DSEC, MVSEC), depth, eye tracking |
| **E. Detection & tracking** | boxes/tracks over time | Yes, and with a temporal-consistency requirement. | Gen1/1Mpx automotive, EBSSA |
| **F. Non-vision event streams** | class per sequence | Yes. | SHD, N-TIDIGITS (audio) |

**In one line: Experiment 1 and 2 ran category A. Experiment 3 moves to category C.**

That is the whole point. Categories A and C look identical from the outside — both are
"classify an event recording into N classes", both load through Tonic, both feed the
same `Conv2d` stack. But A is a static-image problem wearing an event-camera costume,
and C is not. Moving A → C is the smallest possible change that makes the neuron's
memory *load-bearing* rather than decorative — while keeping the task type, the loss,
the metrics and the whole measurement harness exactly as they are.

Categories D and E would also require memory, but they change the output format, the
loss, and every metric in `runs.csv`. They are a different project, not a next step.

### 1.3 What DVS128 Gesture is

Also called the **IBM DVS Gesture dataset**. Introduced by Amir et al. at **CVPR 2017**
in *A Low Power, Fully Event-Based Gesture Recognition System* — the paper that ran
gesture recognition end to end on IBM's TrueNorth neuromorphic chip. It is the de-facto
standard event-camera action-recognition benchmark, and the one both SpikingJelly and
snnTorch ship loaders for.

| | |
|---|---|
| **sensor** | iniLabs DVS128 — **128 × 128** pixels, 2 polarities |
| **classes** | **11** |
| **subjects** | **29** people |
| **lighting conditions** | **3** — natural light, fluorescent, LED |
| **gesture instances** | **1,342** total |
| **duration per instance** | **≈ 2 s to 18 s**, about **6 s** on average |
| **what a sample is** | one person performing one gesture, recorded continuously |

The 11 classes, exactly as Tonic lists them
(**[installed]** `tonic/datasets/dvsgesture.py`):

```
Hand clapping · Right hand wave · Left hand wave
Right arm cw  · Right arm ccw   · Left arm cw · Left arm ccw
Arm roll · Air drums · Air guitar · Other gestures
```

`cw` = clockwise, `ccw` = counter-clockwise. Note **class 11, "Other gestures"** —
subjects were told to invent a gesture of their own. It is deliberately open-ended, and
it is the class everything mis-classifies into. Expect it to dominate the confusion
matrix and say so rather than hiding it.

One important detail about **Tonic's** copy specifically: it is *not* the raw IBM
release. Tonic's own docstring says so —

> *"This is (exceptionally) a preprocessed version of the original dataset, where
> recordings that originally contained multiple labels have already been cut into
> respective samples. Also temporal precision is reduced to ms."*

Two consequences, both good for us and both needing to be stated in the report:
recordings arrive **already cut into one-gesture samples** (no segmentation step to
build), and timestamps are **millisecond**, not microsecond, resolution. Sensor size is
`(128, 128, 2)` and the train/test split is fixed by the dataset, not by us.

### 1.4 The four properties that make this dataset worth the trouble

**(1) The label lives in the temporal order — and nothing else.**

This is the property. The class list contains `Right arm cw` **and** `Right arm ccw`.
It contains `Left arm cw` **and** `Left arm ccw`. It contains `forearm/arm roll` in a
direction. Accumulate any of those into a single picture and you get *the same picture*:
a blurry circle where an arm went round. The classes are separated **only** by the order
in which the pixels lit up.

Formally: the temporal order is **class-conditional**. A model that discards time cannot
exceed chance on those pairs, no matter how large it is. This is exactly what N-MNIST
does not have (§1.5), and it is the reason a neuron's memory can finally be measured
rather than assumed.

**(2) A three-orders-of-magnitude timescale gap.**

Events arrive on a millisecond grid. One gesture cycle — one full arm rotation, one
clap-to-clap interval — takes roughly **0.5 to 1 second**. So the evidence for the label
is spread over ~500–1000 ms, while the frameworks' out-of-the-box neurons remember for
**τ = 1.44 to 9.49 timesteps** (measured in ex2, §3 of `ex2_design.md`).

That gap is precisely what the specialised neurons in §2 exist to close, and it is why
this dataset — not a harder version of N-MNIST — is the right place to test them.

**(3) Genuine sparsity, because the camera does not move.**

The DVS128 was on a tripod. There is no **ego-motion**, so the background emits
essentially nothing and only the moving arm generates events. Two things follow:
the data is genuinely sparse in the way SNN energy arguments assume, and our existing
**spike-rate and energy-proxy metrics measure something meaningful for the first time**
— on N-MNIST the whole frame is in motion during a saccade, so "sparse" was never really
tested.

**(4) The lighting conditions are a built-in nuisance variable.**

Three lighting conditions, one of them **LED**. LED lamps flicker at mains frequency,
and a DVS sees that flicker as a continuous rain of events across the entire sensor.
So one third of the dataset carries structured, periodic noise that the other two thirds
do not. This makes the `Denoise` filter setting matter far more than it did on N-MNIST,
and it gives an honest test of whether a longer-memory neuron is *robust* or merely
*more sensitive*.

### 1.5 N-MNIST vs DVS128 Gesture, side by side

| | **N-MNIST** (ex1, ex2) | **DVS128 Gesture** (ex3) |
|---|---|---|
| category (§1.2) | **A** — converted static | **C** — action recognition |
| what was recorded | a **printed digit**, camera moved in 3 saccades | a **person moving**, camera fixed |
| resolution | 34 × 34 | **128 × 128** (14× the pixels) |
| classes | 10 | 11 |
| duration per sample | **300 ms** | **2–18 s**, ~6 s mean |
| train / test samples | 60,000 / 10,000 | **~1,300 total** (two orders of magnitude fewer) |
| **does collapsing time destroy the label?** | **No** — sum the events and you see the digit | **Yes** — cw and ccw become the same blur |
| is the class a shape or a movement? | a shape | **a movement** |
| camera motion | yes (saccades) → whole frame active | none → only the subject is active |
| noise structure | sensor noise only | sensor noise **+ LED flicker** in 1/3 of samples |
| what the neuron's memory is for | nothing in particular | **the task** |

The row that matters is the bolded one. **Everything ex1 and ex2 measured about
timescales — Norse remembering 6.6× longer, snnTorch being 2× as excitable — was
measured on a dataset where none of it could affect the answer.** ex3 is where those
properties finally have somewhere to show up.

Two rows in that table are the costs, and §5 deals with both: **14× the pixels** changes
the architecture's parameter count (§5.3), and **~1,300 samples instead of 70,000**
changes how much any single run can support (§5.5).

---

## 2. What each framework actually offers

Three separate things must be distinguished, exactly as in ex2:

| label | meaning |
|---|---|
| **(D)** | **constructor default** — what you get with no arguments |
| **(E)** | **the framework's own documented example / reference model** |
| **(—)** | **no framework guidance exists**; we must choose and say so |

The striking finding from reading the three documentation sets is that **all three
frameworks offer a specialised neuron for long-timescale tasks, and all three chose a
different mechanism.** Nobody is copying anybody.

### 2.1 snnTorch 1.0.0 — `snn.Synaptic`

**Mechanism: add a second state variable.**

Official equations, reset-by-subtraction, from the
[Synaptic docs](https://snntorch.readthedocs.io/en/latest/snn.neurons_synaptic.html)
and the installed source:

```
I_syn[t+1] = α · I_syn[t] + I_in[t+1]
U[t+1]     = β · U[t]     + I_syn[t+1]  −  R·U_thr
```

Compare to ex1/ex2's `snn.Leaky`, which is just `U[t+1] = β·U[t] + I_in[t+1]`. The
input no longer lands on the membrane directly. It lands in a **synaptic current**
`I_syn` that has its own decay `α`, and *that* is what drives the membrane. An input
spike now produces a smooth rise-and-fall instead of an instant jump.

Two decays instead of one. Two timescales instead of one. **Two state variables per
neuron instead of one** — which is directly visible in our memory metric.

Constructor, **[installed]** `snntorch/_neurons/synaptic.py`:

```python
def __init__(self, alpha, beta, threshold=1.0, spike_grad=None,
             surrogate_disable=False, init_hidden=False, inhibition=False,
             learn_alpha=False, learn_beta=False, learn_threshold=False,
             reset_mechanism="subtract", state_quant=False, output=False,
             reset_delay=True)
```

| parameter | out-of-box | kind | note |
|---|---|---|---|
| `alpha` | **none** | (—) | **positional, no default.** Same situation as `beta` in ex2: snnTorch ships no synaptic time constant. |
| `beta` | **none** | (—) | positional, no default, as in ex2 |
| `learn_alpha` / `learn_beta` | `False` | (D) | both decays *can* be trained, but are not by default |
| `threshold` | 1.0 | (D) | |
| `reset_mechanism` | `"subtract"` | (D) | soft |
| `reset_delay` | `True` | (D) | reset applied the **following** step |

**Runner-up: `snn.RSynaptic` / `snn.RLeaky`** — the same neuron plus an explicit
recurrent connection from the layer's own output spikes back to its input, with a
learnable weight `V`. And **`snn.SLSTM` / `snn.SConv2dLSTM`**, spiking LSTM cells,
which snnTorch describes for long sequences. Both are further from a "neuron" and
closer to an architecture change, which is why `Synaptic` is the primary candidate:
it changes the neuron and nothing else, keeping the ex1/ex2 comparison structure intact.

Full inventory, **[installed]** `snntorch/_neurons/`: `alpha`, `deltaleaky`, `lapicque`,
`leaky`, `leakykernel`, `leakyparallel`, `leakyunroll`, `linearleaky`, `rleaky`,
`rsynaptic`, `sconv2dlstm`, `slstm`, `stateleaky`, `synaptic`.

### 2.2 SpikingJelly 0.0.0.0.14 — `ParametricLIFNode` (PLIF)

**Mechanism: keep one state variable, but learn its timescale.**

Official equations, from the
[neuron API docs](https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/sub_module/spikingjelly.activation_based.neuron.html):

```
decay_input == True :   H[t] = V[t-1] + (1/τ)·( X[t] − (V[t-1] − V_reset) )
decay_input == False:   H[t] = V[t-1] − (1/τ)·( V[t-1] − V_reset ) + X[t]
```

Identical to `LIFNode`. **One line is different**, and it is the whole idea:

```
1/τ = Sigmoid(w),   w is a learnable parameter
```

τ stops being a hyperparameter you guess and becomes a weight that backpropagation
tunes, one per layer. The `Sigmoid` wrapper is what keeps `1/τ` inside `(0, 1)` so the
decay can never become unstable. Cost: **one extra scalar per layer** — the cheapest of
the three mechanisms by a wide margin.

Constructor, **[installed]** `spikingjelly/activation_based/neuron.py:1013`:

```python
def __init__(self, init_tau: float = 2.0, decay_input: bool = True,
             v_threshold: float = 1., v_reset: float = 0.,
             surrogate_function: Callable = surrogate.Sigmoid(),
             detach_reset: bool = False, step_mode='s', backend='torch',
             store_v_seq: bool = False)
```

| parameter | out-of-box | kind | note |
|---|---|---|---|
| `init_tau` | **2.0** | (D) + (E) | the *initial* τ; it moves during training. Same 2.0 as ex2's `LIFNode`. |
| `decay_input` | `True` | (D) | → gain `1/τ` |
| `v_threshold` / `v_reset` | 1.0 / 0.0 | (D) | hard reset |
| `surrogate_function` | `Sigmoid(4.0)` | (D) | ⚠ its DVS example passes `ATan()` instead — same disagreement as ex2 |
| `detach_reset` | `False` | (D) | ⚠ its DVS example passes `True` — same disagreement as ex2 |

**This is not an obscure corner of the library.** PLIF is Fang et al., **ICCV 2021** —
and it is *the DVS128 Gesture paper*, reporting **97.57%**. SpikingJelly ships the
reference architecture from that paper as a first-class model,
`spikingjelly.activation_based.model.parametric_lif_net.DVSGestureNet`
(**[installed]**), whose file header cites the paper by URL:

```python
DVSGestureNet:  5 × [ Conv2d(·,128,k3,pad1,bias=False) → BatchNorm2d → neuron → MaxPool2d(2) ]
                → Flatten → Dropout(0.5) → Linear(128·4·4, 512) → neuron
                → Dropout(0.5) → Linear(512, 110)  → neuron
                → VotingLayer(10)
```

and the bundled runnable example `examples/classify_dvsg.py` instantiates exactly that
net. Worth noting for §5: the shipped example passes plain `neuron.LIFNode` into a
module family literally named `parametric_lif_net` — so **PLIF vs LIF is a
one-argument swap in SpikingJelly's own example**, which is as clean a control as this
experiment could ask for.

Full inventory, **[installed]** `neuron.py`: `IFNode`, `LIFNode`,
**`ParametricLIFNode`**, `QIFNode`, `EIFNode`, `IzhikevichNode`, `LIAFNode`,
`KLIFNode`, plus the base classes `BaseNode` and `AdaptBaseNode`.

**Runner-up: `KLIFNode`** — a LIF with a learnable non-linearity `k` applied before the
threshold. Same "learn a scalar" philosophy, less well established, and no DVS-Gesture
result behind it.

### 2.3 Norse 1.1.0 — `LSNNCell`

**Mechanism: keep the timescale fixed, and make the threshold depend on firing history.**

LSNN = *Long short-term memory Spiking Neural Network*, from **Bellec et al. 2018**
([arXiv:1803.09574](https://arxiv.org/abs/1803.09574)) — a paper written specifically
about giving SNNs long temporal memory. Norse implements it as a first-class module.

Official equations, verbatim from the installed docstring
(`norse/torch/module/lsnn.py:24`) and the
[LSNN docs](https://norse.github.io/norse/auto_api/norse.torch.module.lsnn.html):

```
v̇ = (1/τ_mem)·(v_leak − v + i)          membrane
i̇ = −(1/τ_syn)·i                        synaptic current
ḃ = −(1/τ_b)·b                          adaptation

z  = Θ( v − v_th + b )                   ← the threshold is v_th + b, not v_th

v ← (1−z)·v + z·v_reset
i ← i + input
b ← b + β·z                              ← every spike raises the bar
```

Read the last line together with the jump condition. Every time the neuron fires, `b`
grows by `β`; and `b` is *added to the threshold*. So a neuron that has just fired needs
more input to fire again, and that penalty decays away on its own slow timescale `τ_b`.
It is a graded, learnable-timescale **refractory period**, and it gives the neuron a
memory of its own recent activity that is completely separate from its membrane.

Three state variables: `v`, `i`, `b`. The most expensive of the three mechanisms.

Defaults, **[installed]** `norse/torch/functional/lsnn.py:37`:

| parameter | value | as a duration (at `dt` = 1 ms) |
|---|---|---|
| `tau_syn_inv` | `1/5e-3` = 200 | τ_syn = **5 ms** |
| `tau_mem_inv` | `1/1e-2` = 100 | τ_mem = **10 ms** |
| `tau_adapt_inv` | `1/800` = 0.00125 | τ_b = **800 ms** |
| `beta` | 1.8 | adaptation strength |
| `v_th` / `v_reset` / `v_leak` | 1.0 / 0.0 / 0.0 | hard reset |
| `method` / `alpha` | `'super'` / 100.0 | ⚠ **alpha is ignored** — the ex2 bug, unchanged |

**The number worth staring at is τ_b = 800 ms against τ_mem = 10 ms — a factor of 80.**
Norse's default adaptation timescale is nearly a second. Compare §1.4(2): a gesture
cycle is 0.5–1 s. Norse's out-of-the-box LSNN is, by pure coincidence of defaults,
tuned to almost exactly the timescale of this dataset. That is a hypothesis waiting to
be tested (H3), not a claim.

**Runner-up: `LIFAdExCell`** — adaptive *exponential* LIF, the biophysical alternative.
Instead of raising the threshold, it grows an inhibitory adaptation current `a`, and it
adds an exponential term that makes spike initiation sharp:

```
v̇ = (1/τ_mem)·( v_leak − v + i + ΔT·exp((v − v_th)/ΔT) )
ȧ = (1/τ_ada)·( a_current·(v − v_leak) − a )
```

Defaults `tau_ada_inv=2.0`, `delta_T=0.5`, `adaptation_current=4.0`,
`adaptation_spike=0.02`
([LIFAdEx docs](https://norse.github.io/norse/auto_api/norse.torch.module.lif_adex.html)).
LSNN is the primary candidate because it is the one with a paper about *temporal memory*
behind it; LIFAdEx is about *neuronal realism*.

Full inventory, **[installed]** `norse/torch/module/`: `coba_lif`, `iaf`, `izhikevich`,
`leaky_integrator`, `leaky_integrator_box`, `lif`, `lif_adex`, `lif_adex_refrac`,
`lif_box`, `lif_correlation`, `lif_ex`, `lif_mc`, `lif_mc_refrac`, `lif_refrac`,
**`lsnn`**. By count, Norse offers the richest neuron zoo of the three.

**And the ex2 caveat still holds:** Norse ships **no event-camera example**
(`norse/task/` = cartpole, cifar10, correlation_experiment, memory, mnist, mnist_pl,
speech_commands). Kind **(E)** does not exist for Norse here either. Its LSNN defaults
are the only guidance it gives.

### 2.4 The choice, in one table

| | snnTorch | SpikingJelly | Norse |
|---|---|---|---|
| **primary** | `snn.Synaptic` | `neuron.ParametricLIFNode` | `LSNNCell` |
| **runner-up** | `snn.RSynaptic`, `snn.SLSTM` | `neuron.KLIFNode` | `LIFAdExCell` |
| mechanism | second state variable | learned time constant | spike-driven adaptive threshold |
| the slogan | *add a timescale* | *learn the timescale* | *make the threshold remember* |
| provenance | classic 2nd-order LIF | Fang et al., ICCV 2021 | Bellec et al., NeurIPS 2018 |
| has a published DVS-Gesture result? | no | **yes, 97.57%** | no |

---

## 3. The three mechanisms on one scale

Converting §2 into numbers that can be compared across frameworks.
**Not yet measured** — unlike ex2's §3 table, these are read from the documented
equations and defaults. Measuring them (the ex2 approach: build each neuron, raise the
threshold out of reach, drive it, record) is step 1 of §8.

| | snnTorch `Synaptic` | SpikingJelly `PLIF` | Norse `LSNN` |
|---|---|---|---|
| **state variables per neuron** | **2** (`I_syn`, `U`) | **1** (`V`) | **3** (`v`, `i`, `b`) |
| **extra trainable parameters** | 0 by default (2 per layer if `learn_alpha`/`learn_beta`) | **1 per layer** (`w`, giving τ) | 0 by default |
| **timescales available** | 2, both fixed by you | 1, **learned** | 3, fixed by you (5 ms / 10 ms / **800 ms**) |
| **longest default timescale** | none — no defaults exist | τ starts at 2.0 steps and moves | **τ_b = 800 ms** |
| **is the long timescale learned?** | no | **yes** | no |
| **is the long timescale activity-dependent?** | no | no | **yes** — driven by the neuron's own spikes |
| **memory cost vs plain LIF** | ×2 | ×1 (+1 scalar/layer) | **×3** |
| reset | soft, delayed 1 step | hard | hard |
| surrogate | atan(2.0) *(if we keep ex2's)* | ATan/Sigmoid | super → **effectively α=1** (bug) |

**Three things fall out, and they become hypotheses in §7:**

**1. The cost ordering is unambiguous and measurable.** State variables per neuron are
1 : 2 : 3 for PLIF : Synaptic : LSNN. Our existing memory metric measures exactly this,
so for the first time in this project a *predicted* memory ranking exists before the run.

**2. Only one of the three closes the timescale gap out of the box.** §1.4(2) says the
task needs ~0.5–1 s of memory. Norse's LSNN has τ_b = 800 ms sitting in its defaults.
snnTorch has no defaults at all, so whatever it gets is our choice. SpikingJelly starts
at τ = 2 steps and has to *learn* its way up.

**3. Only one of the three can adapt per-sample.** PLIF learns one τ per layer during
training and then freezes it at inference. LSNN's `b` changes *within a single sample*,
in response to that sample's own activity. That is a qualitatively different kind of
memory, and on a dataset with 2 s and 18 s samples in the same batch it may matter.

---

## 4. Why there is no equivalence check this time

Experiments 1 and 2 both ran `equivalence_check.py` first, because in both cases the
three frameworks were computing *the same neuron* — forced (ex1, agreeing to 1.19e-07)
or defaulted (ex2, diverging to ~2.3e-01). Either way the question "do these membrane
traces match?" was meaningful.

**Here it is not.** A 1-state learned-τ LIF, a 2-state synaptic-current LIF and a
3-state adaptive-threshold LIF do not have comparable membrane traces. Asking whether
they agree is asking whether an apple weighs the same as Tuesday. Running the script and
reporting a big number would be theatre.

Two things replace it:

**(a) A shared plain-LIF control arm.** Run ex2's out-of-the-box `Leaky` / `LIFNode` /
`LIFBoxCell` on DVS128 Gesture first, unchanged. That arm *does* support the ex1/ex2
equivalence machinery, it anchors this dataset against the two experiments we already
have, and it is the baseline every specialised neuron is measured against. §5.1.

**(b) A characterisation probe instead of a comparison.** Rather than "do they agree",
ask "what does each one *do*" — build one of each, drive it with a step and a Poisson
train, and record: impulse response, effective longest timescale, adaptation decay,
and firing rate under sustained drive. Same `equivalence_check.py` scaffolding, no
verdict, three separate characters instead of one shared trace. This is the ex3 analogue
of ex2's §3 table, and it is the honest version of the question.

---

## 5. Scope: what changes, and what deliberately does not

### 5.1 The decision — two arms, one factor each

Everything in this project so far has changed exactly one thing at a time. ex3 changes
two (dataset **and** neuron), so it needs two arms to stay interpretable:

| arm | dataset | neuron | what it isolates |
|---|---|---|---|
| **ex3a — control** | DVS128 Gesture | ex2's out-of-the-box plain LIF, **unchanged** | **the dataset alone.** Same neuron as ex2 → any change vs ex2 is the dataset's doing. |
| **ex3b — treatment** | DVS128 Gesture | each framework's specialised neuron (§2.4) | **the neuron alone.** Same dataset as ex3a → any change vs ex3a is the neuron's doing. |

Without ex3a there is no way to attribute anything: a big accuracy jump could be the
specialised neuron or could be that this dataset simply behaves differently. ex3a costs
one extra seed sweep and makes every number in ex3b interpretable. **It is not optional.**

`ex3a` doubles as the sanity check that the whole DVS-Gesture pipeline works, before any
new neuron code is trusted.

### 5.2 Why not each framework's whole recipe (again)

Same argument as ex2 §5.2, and it applies with more force here because SpikingJelly's
DVS-Gesture recipe is genuinely large:

| | SpikingJelly `classify_dvsg.py` | ex1/ex2/ex3 shared |
|---|---|---|
| architecture | 5×(128C3-BN-neuron-MP2), FC512, FC110, VotingLayer | 12C5-MP2-32C5-MP2-FC |
| channels | 128 | 12 / 32 |
| normalisation | BatchNorm2d after every conv | none |
| regularisation | Dropout 0.5 ×2 | none |
| output | 110 neurons → VotingLayer(10) | one neuron per class |
| T | 16 | ours |
| batch / lr / epochs | 16 / 0.1 / 64 | 128 / 0.002 / 5 |

Adopting that for SpikingJelly and not the others would compare training recipes, not
frameworks — and no column in `runs.csv` would mean the same thing in two rows.
**Keep the shared architecture.** Its provenance (snnTorch Tutorial 7, verified in ex2)
is unchanged, and the shared-everything-but-one-thing rule is the only reason any of the
previous numbers were comparable.

### 5.3 The architecture problem this dataset creates — needs your decision

This is the one genuinely new engineering issue, and it is a consequence of §1.5's
"14× the pixels" row. The shared architecture applied to 128 × 128:

```
input            2 × 128 × 128
Conv2d(2→12,k5)  12 × 124 × 124
LIF              12 × 124 × 124
MaxPool2d(2)     12 ×  62 ×  62
Conv2d(12→32,k5) 32 ×  58 ×  58
LIF              32 ×  58 ×  58
MaxPool2d(2)     32 ×  29 ×  29
Flatten               26,912          ← was 800 on N-MNIST
Linear(26912→11)                      ← 296,043 parameters
```

| | N-MNIST (ex1/ex2) | DVS Gesture, unchanged architecture |
|---|---|---|
| flatten width | 800 | **26,912** (34×) |
| total parameters | 18,254 | **≈ 306,000** (17×) |
| share of parameters in the final `Linear` | 44% | **97%** |

That last row is the problem. The network stops being "a spiking conv net" and becomes
"a large linear classifier with a small spiking front end" — and a linear classifier
reading a flattened spike-count vector is exactly the kind of model that can ignore
temporal structure. On a dataset chosen *because* temporal structure is the label,
that would undercut the whole experiment. On top of that, ~1,300 training samples
against 306k parameters is a serious over-fitting risk (§8.6).

`infer_flat_features()` handles this automatically — nothing will crash. The question is
scientific, not technical.

Three options, and **this one is yours to decide**:

| option | how | pro | con |
|---|---|---|---|
| **A. Downsample the events to 32 × 32 or 64 × 64** | `tonic.transforms.Downsample(spatial_factor=0.25)` before `ToFrame` (**[installed]**, tonic 1.6.0) | architecture and parameter count stay close to ex1/ex2, so cross-experiment comparison survives; one line of config | throws away spatial detail; a preprocessing choice we must justify |
| **B. Add a third conv+pool block** | one line in `build_feature_layers` | keeps full resolution, brings flatten to ~32·13·13 = 5,408 | changes the architecture, so ex1/ex2 comparisons weaken |
| **C. Change nothing** | — | zero new decisions | 97% of parameters in one linear layer; over-fitting; weakens the temporal argument |

**Recommendation: A, at `spatial_factor=0.25` → 32 × 32.** It keeps the architecture
byte-identical to ex1 and ex2 (34×34 → 32×32 is a 2-pixel difference, flatten 800 → 512),
so the whole three-experiment series stays on one architecture, and the change is a
*data* transform which is applied identically to all three frameworks and therefore
cannot bias the comparison. It also shrinks the cache and the runtime, which matters on
free Colab. But it is a real loss of information and the report must say so.

### 5.4 T and framing — the other decision that is yours

On N-MNIST, T = 20 over a 300 ms sample means each frame is 15 ms. On DVS Gesture with
`n_time_bins`, T = 20 over a **6 s** sample means each frame is **300 ms** — and over an
18 s sample, 900 ms. Two problems at once:

- **The frames get coarse.** A 300 ms bin may contain an entire quarter-rotation of an
  arm, blurred into one picture. Temporal detail is destroyed *before* the neuron sees it.
- **`n_time_bins` makes T sample-length-invariant**, so a 2 s and an 18 s gesture both
  become 20 frames — meaning the *physical* duration of one timestep differs by 9× within
  a single batch. Every neuron time constant is expressed in timesteps, so the same τ
  means different things for different samples.

The alternative, `time_window` framing, gives every frame the same real duration and pads
or crops to a fixed count — `PadOrCropFrames` already exists in `src/data.py` for exactly
this. It is more physically meaningful and it is what makes "τ = 800 ms" a statement you
can check.

Both modes already work in the config. **The pair (framing mode, T) is a values decision
and therefore yours.** The relevant facts, stated without a recommendation:

- SpikingJelly's own example uses **T = 16** with `frames_number=16, split_by='number'`
  (event-count binning, a third mode we do not currently support).
- The literature on this dataset commonly uses T = 16 or T = 20.
- Larger T costs linearly in time and memory: T = 60 is 3× the training time of T = 20.
- Under `time_window`, T × window must cover enough of the gesture to contain the
  discriminative motion.

### 5.5 The two dataset facts that will show up in the results

Flagging these now so they are not surprises later.

**~1,300 samples, not 70,000.** N-MNIST's test set has 10,000 samples, so 0.1% accuracy
resolution. DVS Gesture's test split has a few hundred, so **one test sample is worth
roughly 0.3 percentage points**. Accuracy will be visibly chunky, and ex1's ~0.15 pp
noise floor **does not transfer** — it must be re-established on this dataset from the
three seeds before any accuracy claim is made. This is the single most likely way to
publish a false ranking here.

**"Other gestures" is an open class.** §1.3. Expect it to be the worst class and the
sink for most confusion. A per-class breakdown or confusion matrix is worth adding to
the report for this dataset in a way it never was for N-MNIST.

---

## 6. What stays forced, and why

Carry-overs that are **not** out-of-the-box and must be declared, same as ex2 §6:

| forced | why |
|---|---|
| **SpikingJelly runs `step_mode='s'`, `backend='torch'`** | The shared network loops over timesteps. Its fused CUDA kernel needs `'m'` and the whole `[T,…]` tensor. Verified in ex1 that `'m'` and looped `'s'` give identical spikes → speed-only limitation. Same as ex1 (L2) and ex2. **Note this now interacts with open item E5.** |
| **Norse uses `circ(0.5)`, not the default `super`** | Norse's default SuperSpike silently ignores its `alpha` (ex2 §2.3, `local_docs/norse_superspike_alpha_finding.md`), behaving as α=1 with ~2.7× fatter gradient tails at \|v−θ\|=1. On a task requiring credit assignment across many more timesteps than N-MNIST, that mis-scaling compounds further. ex2 measures the bug deliberately; **ex3 should control it away** so that any Norse result is about LSNN, not about a released bug. |
| **The shared architecture** | §5.2. |
| **One shared learning rate** | ex1 limitation L7, restated in ex2 §5.4. The risk is *larger* again here, because the three neurons now have structurally different gradient paths (2 states, 1 learned scalar, 3 states). Keep it shared — a per-framework lr reintroduces the recipe problem — but expect to state that a framework may be underperforming on gradient scale rather than on merit. The `equivalence_check.py` gradient-norm probe is what would settle it. |

One genuinely new asymmetry, which must be stated plainly:

**snnTorch's `Synaptic` has no defaults for `alpha` or `beta`, and no event-data example
using it.** For SpikingJelly and Norse, the specialised neuron comes with the framework's
own numbers. For snnTorch, we have to pick both time constants ourselves. That is kind
**(—)**, it is a real finding about snnTorch (the same finding as ex2's missing `beta`,
now doubled), and the values chosen must be labelled as ours, not the framework's.

---

## 7. What to expect — hypotheses, not conclusions

Written in advance so the results can contradict them. Derived from the documentation
read in §2 and the dataset properties in §1.4 — none of these has been measured.

| # | hypothesis | reasoning | measured by |
|---|---|---|---|
| **H1** | **The plain-LIF control arm (ex3a) scores far below its N-MNIST accuracy — a much bigger drop than the change in class count explains.** | §1.4(2): out-of-the-box τ is 1.44–9.49 timesteps, and the label is spread over the whole sample. A neuron that remembers 2 frames cannot represent a direction of rotation. | ex3a accuracy vs ex2's |
| **H2** | **Within each framework, the specialised neuron beats the plain LIF by more than that framework's own noise floor.** | Each mechanism in §2 adds exactly what H1 says is missing. If this fails for all three, the finding is that the neuron is not the bottleneck — which is also a result. | ex3b − ex3a, per framework, F6.1 effect size |
| **H3** | **The size of that gain differs by mechanism, and Norse's LSNN gains the most out of the box.** | §3(2): only LSNN ships a default timescale (τ_b = 800 ms) matched to the gesture timescale. PLIF must learn its way there from τ = 2 steps in 5 epochs; snnTorch's is whatever we pick. **The falsifiable part is the ordering, not the magnitudes.** | ex3b ranking |
| **H4** | **Memory cost follows state count: PLIF ≈ plain LIF < Synaptic < LSNN, roughly 1 : 2 : 3 per neuron.** | §3 table. This is the most nearly *deterministic* prediction in the list — if the measurement contradicts it, the measurement is wrong or the framework is doing something undocumented. | existing peak-memory metric |
| **H5** | **LSNN has the lowest spike rate; Synaptic the highest.** | Adaptation directly suppresses repeat firing (`b += β·z` raises the bar). Synaptic current spreads one input event across several timesteps, so more timesteps sit near threshold. | existing spike-count metric, and M2 (SOPs) if implemented |
| **H6** | **PLIF gives the best accuracy-per-unit-cost.** | One learnable scalar per layer, no extra state, and a published 97.57% on this exact dataset. If H3 and H6 both hold, the story is "Norse wins out of the box, SpikingJelly wins per joule", which is a genuinely useful finding. | accuracy vs memory / energy / spike count |
| **H7** | **Between-framework accuracy differences here exceed ex2's.** | ex1: same neuron, differences = noise. ex2: same neuron *class*, different parameters. ex3: structurally different neurons. Each step should widen the spread. **But §5.5's small test set means the noise floor must be re-established first** — this hypothesis is untestable until it is. | F2.2 paired differences, F6.1 effect size |
| **H8** | **Speed rankings move, unlike ex2's H5.** | ex2 predicted stable rankings because the arithmetic per neuron was unchanged. That is false here: Synaptic and LSNN add elementwise ops and state tensors per timestep, PLIF adds a sigmoid. If a ranking *doesn't* move, that is interesting and needs explaining. | existing wall-clock and latency metrics |

**The two hypotheses that make this experiment worth running are H2 and H6.**
H2 asks the plain question — *does the framework's own specialised neuron actually help
on the task it was built for?* — which, surprisingly, is not something the literature
answers for all three frameworks under one harness. H6 asks the question a practitioner
actually has: *what does that help cost?* Everything else is supporting evidence.

**And one honest caveat about the whole set:** ex1 and ex2 could hold the neuron constant
and vary only the framework. ex3 cannot — the specialised neurons *are* the frameworks'
differences. So ex3 does not measure "which framework is better". It measures
**what each framework hands you when you follow its own advice for a temporal task, and
what that choice costs.** That is a narrower claim than ex1's, and the report must make
the narrowing explicit rather than letting the reader over-read the ranking.

---

## 8. Concrete plan

### 8.1 Code changes — unlike ex2, there are some

ex2 needed none. ex3 needs four small, separable pieces. None of them touch the training
loop, the metrics, the results writer or the plotting.

**(1) `src/data.py` — add DVS128 Gesture.** Two edits:

```python
DATASETS = {
    "nmnist":     {"class": tonic.datasets.NMNIST,     "num_classes": 10},
    "dvsgesture": {"class": tonic.datasets.DVSGesture, "num_classes": 11},   # NEW
}
```

and `build_split()` currently hardcodes N-MNIST's raw-event options:

```python
dataset_options = {
    "first_saccade_only": require_bool(dataset_cfg, "first_saccade_only"),
    "stabilize":          require_bool(dataset_cfg, "stabilize"),
}
```

`tonic.datasets.DVSGesture.__init__` accepts neither (**[installed]**, its signature is
`save_to, train, transform, target_transform, transforms`). So these must become
per-dataset — e.g. a `"options"` list in the `DATASETS` entry, read only if present.
Keep them in the cache key exactly as now, so cache identity is unaffected.
This is the only change that touches shared code, and it is additive.

**(2) Optional spatial downsample** (§5.3 option A) — one stage in the transform chain,
before `Denoise`, gated on a config key. It changes the events, so it **must** join the
cache identity (`cache_key`), or a 32×32 cache could be silently read as 128×128.

**(3) Three new adapters** in `src/adapters/`, alongside the existing three:

| file | class | notes |
|---|---|---|
| `snntorch_synaptic.py` | `SnnTorchSynaptic` | returns `(spk, syn, mem)` instead of `(spk, mem)` — the adapter holds two state tensors |
| `spikingjelly_plif.py` | `SpikingJellyPLIF` | near drop-in; `ParametricLIFNode` has the same interface as `LIFNode`. **Its learnable `w` is a module parameter, so `build_network`'s "LIF wrappers draw no random numbers" invariant needs re-checking** — if PLIF's `w` is initialised deterministically from `init_tau` (it should be), the identical-weights guarantee survives; verify, don't assume. |
| `norse_lsnn.py` | `NorseLSNN` | state is an `LSNNFeedForwardState` NamedTuple, handled exactly like the existing `LIFBoxFeedForwardState` — the current Norse adapter's `None`-means-fresh pattern carries over unchanged |

All three implement the same `BaseLIF` interface the existing adapters do
(`forward`, `reset`, `has_state`, `describe`). **`src/network.py` needs no change** —
it only ever calls `make_lif()` and treats the result as a layer. The design note in its
docstring ("Only `make_lif` changes between frameworks") is about to be tested properly,
which is itself worth a line in the report.

**(4) `describe()` must report the new parameters** so they land in the run's config
snapshot: `alpha`/`beta` for Synaptic, the **learned** τ for PLIF (before *and after*
training — see below), `tau_adapt_inv`/`beta` for LSNN.

**One new measurement is free and should not be skipped: PLIF's learned τ.** It starts at
2.0 and training moves it. Reading `Sigmoid(w)` per layer after training costs nothing
and answers §3's open question directly — *did it learn its way to the gesture
timescale?* If PLIF's τ climbs and accuracy climbs with it, that is the cleanest single
figure this experiment could produce. Log it per epoch if cheap.

### 8.2 Config sketch

Two new files, both extending `config/default.yaml` the way `config_ex2.yaml` does.

`config/config_ex3a.yaml` — control arm. Identical to `config_ex2.yaml` except the
`dataset:` block:

```yaml
extends: config/config_ex2.yaml     # ex2's out-of-the-box neurons, unchanged

dataset:
  name: dvsgesture
  root: data
  cache_dir: cache
  denoise_filter_time_us: 10000     # carried from ex1/ex2 — but see §1.4(4): LED
                                    # flicker may justify revisiting this
  binarize: false                   # carried
  downsample_spatial_factor: 0.25   # §5.3 option A -> 32x32   [YOUR DECISION]
  framing:
    mode: n_time_bins               # or time_window            [YOUR DECISION]
    n_time_bins: 20                 #                           [YOUR DECISION]
  batch_size: 128                   # may need lowering: 128x128 frames are 14x
                                    # bigger than 34x34, so T x B x 2 x H x W grows
  num_workers: 2
```

`config/config_ex3b.yaml` — treatment arm. Same dataset block, new `neuron:` block:

```yaml
extends: config/config_ex3a.yaml

neuron:
  snntorch:
    class: synaptic
    alpha: ???                # (—) NO DEFAULT EXISTS. Yours to choose. §6
    beta:  ???                # (—) NO DEFAULT EXISTS. Yours to choose. §6
    threshold: 1.0            # (D)
    reset_mechanism: subtract # (D) soft
    reset_delay: true         # (D)
    surrogate: { type: atan, alpha: 2.0 }

  spikingjelly:
    class: parametric_lif
    init_tau: 2.0             # (D) — the STARTING value; training moves it
    decay_input: true         # (D)
    v_threshold: 1.0          # (D)
    v_reset: 0.0              # (D) hard
    detach_reset: true        # (E) its own DVS example
    step_mode: s              # forced, §6
    backend: torch            # forced, §6
    surrogate: { type: atan, alpha: 2.0 }   # (E)

  norse:
    class: lsnn
    dt: 0.001                 # (D)
    tau_syn_inv: 200.0        # (D) -> 5 ms
    tau_mem_inv: 100.0        # (D) -> 10 ms
    tau_adapt_inv: 0.00125    # (D) -> 800 ms   <- the interesting one, §3
    beta: 1.8                 # (D) adaptation strength
    v_th: 1.0                 # (D)
    v_reset: 0.0              # (D)
    v_leak: 0.0               # (D)
    surrogate: { type: circ, alpha: 0.5 }   # forced, §6 — NOT the default 'super'
```

The `class:` key is new and is what selects the adapter. Everything else follows the
existing per-framework block convention exactly.

### 8.3 Run order

```powershell
# 0. Characterise the three specialised neurons locally, no GPU (§4b).
#    Do this BEFORE any training: it is the ex3 analogue of ex2's section 3 table,
#    and it is what makes the results interpretable.
.venv\Scripts\python equivalence_check.py --config config/config_ex3b.yaml --experiment ex3

# 1. Build the data cache once. DVS Gesture frames are 14x bigger than N-MNIST's
#    (or equal, with 32x32 downsampling) -- check the cache size before committing
#    to a Colab session.
.venv\Scripts\python prepare_data.py --config config/config_ex3a.yaml

# 2. Weight-init check: all three must still start byte-identical.
#    Especially important now -- PLIF adds a parameter (see 8.1 note 3).
.venv\Scripts\python check_network.py --all --config config/config_ex3b.yaml

# 3. ex3a CONTROL: 3 frameworks x 3 seeds, plain LIF on DVS Gesture
python train.py --config config/config_ex3a.yaml --experiment ex3a \
    --framework <fw> --seed <s> --results-root <drive>

# 4. ex3b TREATMENT: 3 frameworks x 3 seeds, specialised neurons
python train.py --config config/config_ex3b.yaml --experiment ex3b \
    --framework <fw> --seed <s> --results-root <drive>

# 5. Home, then plots
python collect_results.py --from <folder>/ex3a --experiment ex3a
python collect_results.py --from <folder>/ex3b --experiment ex3b
python make_plots.py --experiment ex3
```

**18 runs, not 9.** Two arms × 3 frameworks × 3 seeds. Three seeds remain mandatory —
ex1 showed one seed publishes false rankings, and §5.5 says the small test set makes that
*more* likely here, not less. Across several free Colab accounts this is roughly two
sessions' work; plan it with `experiments/exN/` and `collect_results.py` as usual.

### 8.4 What the existing tooling gives for free

- **`make_plots.py` works unchanged.** The plotting schema's condition axis is generic,
  so ex3a-vs-ex3b is just another condition pair.
- **F6.1 (effect size vs noise) is the figure that decides H2 and H7** — but only after
  §5.5's re-established noise floor is fed into it. Do not reuse ex1's 0.15 pp.
- **F2.2 (paired differences)** decides whether any new ranking is real or a swap.
- The memory metric already resolves H4 with no new work.
- **Worth adding for this dataset only:** a per-class confusion matrix (§5.5), and the
  PLIF learned-τ trace (§8.1).

### 8.5 Deliverables

`experiments/ex3/` with `report_ex3.md`, `characterisation/`, `results/`, `figures/` —
same layout as ex1 and ex2, same report structure. Results for both arms in one report,
because neither arm means anything alone.

### 8.6 Risks, named in advance

| risk | why it is real here | mitigation |
|---|---|---|
| **Over-fitting** | ~1,300 training samples against ≥18k parameters (306k if §5.3 option C). N-MNIST had 60,000. | §5.3 option A shrinks the model; watch train-vs-test accuracy every epoch; consider more epochs but report the gap |
| **The noise floor swamps the effect** | §5.5 — one test sample ≈ 0.3 pp | re-establish the floor from the 3 seeds *before* claiming any ranking; this is a reporting discipline, not a code change |
| **Cache size / Colab disk** | 128×128 × T × 2 channels × ~1,300 samples | measure the cache after `prepare_data.py` on a few samples before committing; §5.3 option A also solves this |
| **5 epochs is not enough for PLIF** | PLIF must *learn* τ from 2.0. ex1's 5 epochs were tuned for N-MNIST convergence, and this is a harder, smaller dataset. | log the learned τ per epoch (§8.1); if it is still moving at epoch 5, the epoch budget is the confound, not the neuron |
| **snnTorch's undefined `alpha`/`beta`** | §6 — we choose them, so a bad choice looks like a framework weakness | choose them by the §4b characterisation, document the reasoning, and consider a small sweep as a separate note |
| **Batch size 128 may not fit** | frames are 14× bigger | if it must drop, it must drop **for all three frameworks equally** — a per-framework batch size destroys the speed comparison (ex1 L2 territory) |

---

## 9. Sources

Everything above traces to one of these. Marked **[installed]** where the claim was
verified by reading the installed package (authoritative for the exact versions we run)
rather than a web page.

### Framework documentation

| # | source |
|---|---|
| T1 | **snnTorch neuron index** — the full class list (`Leaky`, `RLeaky`, `Synaptic`, `RSynaptic`, `Lapicque`, `Alpha`, `LeakyParallel`, `SLSTM`, `SConv2dLSTM`). https://snntorch.readthedocs.io/en/latest/snntorch.html |
| T2 | **snnTorch `snn.Synaptic`** — both reset equations, α = synaptic current decay, β = membrane decay, the full constructor signature, `learn_alpha`/`learn_beta` defaulting to False, and **α and β having no defaults**. https://snntorch.readthedocs.io/en/latest/snn.neurons_synaptic.html · also **[installed]** `snntorch/_neurons/synaptic.py:156` |
| T3 | **snnTorch neuron inventory** — **[installed]** `snntorch/_neurons/`: alpha, deltaleaky, lapicque, leaky, leakykernel, leakyparallel, leakyunroll, linearleaky, rleaky, rsynaptic, sconv2dlstm, slstm, stateleaky, synaptic |
| T4 | **SpikingJelly `ParametricLIFNode`** — both `decay_input` equations, `1/τ = Sigmoid(w)` with `w` learnable, and the paper reference. https://spikingjelly.readthedocs.io/zh-cn/0.0.0.0.14/sub_module/spikingjelly.activation_based.neuron.html · also **[installed]** `spikingjelly/activation_based/neuron.py:1013` |
| T5 | **SpikingJelly neuron inventory** — **[installed]** `neuron.py`: BaseNode, AdaptBaseNode, IFNode, LIFNode, ParametricLIFNode, QIFNode, EIFNode, IzhikevichNode, LIAFNode, KLIFNode |
| T6 | **SpikingJelly `DVSGestureNet`** — the reference architecture from the PLIF paper, 5×(Conv3-BN-neuron-MP2) → FC512 → FC110 → VotingLayer(10); file header cites arXiv 2007.05785. **[installed]** `spikingjelly/activation_based/model/parametric_lif_net.py:122` |
| T7 | **SpikingJelly `classify_dvsg.py`** — its runnable DVS128 Gesture example: `DVSGestureNet(channels, spiking_neuron=neuron.LIFNode, surrogate_function=surrogate.ATan(), detach_reset=True)`, `frames_number=args.T, split_by='number'`, optional cupy backend. **[installed]** `spikingjelly/activation_based/examples/classify_dvsg.py:39` |
| T8 | **Norse `LSNN`** — LSNNCell / LSNNRecurrentCell / LSNN / LSNNRecurrent, all three ODEs, the jump condition `z = Θ(v − v_th + b)`, and the transition `b ← b + βz`. https://norse.github.io/norse/auto_api/norse.torch.module.lsnn.html · also **[installed]** `norse/torch/module/lsnn.py:24` |
| T9 | **Norse `LSNNParameters` defaults** — `tau_syn_inv = 1/5e-3`, `tau_mem_inv = 1/1e-2`, `tau_adapt_inv = 1/800`, `beta = 1.8`, `method='super'`, `alpha=100`. **[installed]** `norse/torch/functional/lsnn.py:37` |
| T10 | **Norse `LIFAdEx`** — the adaptive-exponential equations and defaults (`tau_ada_inv=2.0`, `delta_T=0.5`, `adaptation_current=4.0`, `adaptation_spike=0.02`); adapted from Scholarpedia's AdEx article. https://norse.github.io/norse/auto_api/norse.torch.module.lif_adex.html |
| T11 | **Norse module inventory** — https://norse.github.io/norse/auto_api/norse.torch.module.html · also **[installed]** `norse/torch/module/`: coba_lif, iaf, izhikevich, leaky_integrator, leaky_integrator_box, lif, lif_adex, lif_adex_refrac, lif_box, lif_correlation, lif_ex, lif_mc, lif_mc_refrac, lif_refrac, lsnn |
| T12 | **Norse bundled tasks** — cartpole, cifar10, correlation_experiment, memory, mnist, mnist_pl, speech_commands. **Still no event-camera task.** **[installed]** `norse/task/` |

### Dataset

| # | source |
|---|---|
| T13 | **Tonic `DVSGesture`** — the 11 class names, `sensor_size = (128, 128, 2)`, the Amir et al. 2017 citation, and the note that Tonic's copy is pre-cut into single-gesture samples at millisecond precision. **[installed]** `tonic/datasets/dvsgesture.py` · https://tonic.readthedocs.io/en/latest/generated/tonic.datasets.DVSGesture.html |
| T14 | **IBM DVS Gesture project page** — the original release. http://research.ibm.com/dvsgesture/ |
| T15 | **Tonic transforms** — `Downsample`, `Denoise`, `ToFrame`, `CropTime` and the rest, tonic 1.6.0. **[installed]** · https://tonic.readthedocs.io/en/latest/reference/transformations.html |
| T16 | **Neuromorphic Systems dataset card** — the 1,342-instance count, 2–18 s duration range and ~6 s mean. https://neuromorphicsystems.github.io/land/dvs-gesture |

### Publications

| # | source |
|---|---|
| T17 | Amir A., Taba B., Berg D., Melano T., McKinstry J., Di Nolfo C., Nayak T., Andreopoulos A., Garreau G., Mendoza M. et al. (2017), *A Low Power, Fully Event-Based Gesture Recognition System*, **CVPR 2017**, pp. 7243–7252. The dataset paper: 11 gestures, 29 subjects, 3 lighting conditions, DVS128 sensor, TrueNorth deployment. https://openaccess.thecvf.com/content_cvpr_2017/html/Amir_A_Low_Power_CVPR_2017_paper.html |
| T18 | Fang W., Yu Z., Chen Y., Masquelier T., Huang T., Tian Y. (2021), *Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks*, **ICCV 2021**, pp. 2661–2671. The PLIF paper: `1/τ = Sigmoid(w)`, `init_tau = 2.0`, and **97.57% on DVS128 Gesture**. arXiv:2007.05785 · https://openaccess.thecvf.com/content/ICCV2021/papers/Fang_Incorporating_Learnable_Membrane_Time_Constant_To_Enhance_Learning_of_Spiking_ICCV_2021_paper.pdf |
| T19 | Bellec G., Salaj D., Subramoney A., Legenstein R., Maass W. (2018), *Long short-term memory and Learning-to-learn in networks of spiking neurons*, **NeurIPS 2018**. The LSNN paper Norse implements — adaptive threshold for long temporal dependencies. arXiv:1803.09574 · https://arxiv.org/abs/1803.09574 |

### Our own prior work

| # | source |
|---|---|
| T20 | `experiments/ex1/report_ex1.md` — the ~0.15 pp N-MNIST accuracy noise floor (**which §5.5 says does NOT transfer to this dataset**), the forced-neuron configuration, and limitations L2 and L7, both of which carry into ex3. |
| T21 | `experiments/ex2/ex2_design.md` — the out-of-the-box neuron table (decay, gain, DC gain, **τ = 1.44–9.49 timesteps**, which is the number H1 rests on), the (D)/(E)/(—) convention reused here, and the argument in its §5.2 against adopting whole recipes. |
| T22 | `local_docs/norse_superspike_alpha_finding.md` — the SuperSpike `alpha` bug: byte-identical gradients for α = 1/10/100, measured 6.03× gradient-norm ratio, upstream commit `1d2671a` unreleased. The reason §6 forces `circ(0.5)` here. |
| T23 | `docs/open_items.md` — **E3** (second dataset, DVS128 Gesture named as the intended one) and **E4** (other neuron types), which this experiment merges. Also **E5** (SpikingJelly multi-step cupy), **M2** (SOPs) and **M3** (per-timestep spike rates), all of which would strengthen ex3 if done first. |

---

## 10. Open questions for you

Ordered by how much they change the experiment. The first three are values decisions and
are yours by default; the last two are scope.

1. **§5.3 — the resolution.** 128 × 128 unchanged puts 97% of the parameters in one
   linear layer, on a dataset with ~1,300 training samples. Recommendation is
   `Downsample(0.25)` → 32 × 32, which keeps the architecture identical to ex1 and ex2.
   Accept, or prefer a third conv block, or keep full resolution and accept the risk?

2. **§5.4 — framing mode and T.** `n_time_bins` makes T constant but lets one timestep
   mean 300 ms on a short gesture and 900 ms on a long one. `time_window` fixes the
   physical duration of a timestep, which is what makes "τ_b = 800 ms" checkable, but
   pads and crops instead. Both already work. Which, and at what T?

3. **§6 — snnTorch's `alpha` and `beta`.** No defaults exist and no snnTorch event
   example uses `Synaptic`, so both numbers are ours. Pick them from the §4b
   characterisation so they are at least principled, or is there a source you would
   rather cite?

4. **§5.1 — is the control arm worth 9 extra runs?** Recommendation is yes, emphatically:
   without ex3a there is no way to separate "the neuron helped" from "this dataset just
   behaves differently", and every number in ex3b becomes an anecdote. But it doubles the
   Colab budget, so it should be a conscious choice.

5. **Sequencing against `docs/open_items.md`.** E5 (SpikingJelly multi-step + cupy) is
   currently queued as "next after ex2". Running it *before* ex3 would mean this
   experiment measures SpikingJelly at full speed rather than at a lower bound —
   relevant because H8 predicts speed rankings move. Alternatively M2 (SOPs) is
   analysis-only, needs no new runs, and would make H5 much stronger. Do either of these
   jump the queue?
