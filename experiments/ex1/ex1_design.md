# Experiment 1 — design notes: adding sinabs

**Status:** wiring done and config-verified · **sinabs is NOT installed yet, so nothing has been run**
**Written:** 2026-08-12
**Scope of this document:** how the **sinabs** neuron is configured to match ex1's
existing target neuron. It does **not** restate ex1's dataset, architecture,
training recipe or metrics — those live in `config/default.yaml` and
`experiments/ex1/report_ex1.md`.

**Background reading, if any of the terms below are unfamiliar:**
`local_docs/SNNs_Introduction_BaseConcepts.md` (α, τ, DC gain, ReLU, IAF vs LIF)
and `local_docs/Intro-to-Sinabs.md` (what sinabs is, its full parameter list,
every quoted source line).

---
---

# Part 1 — the setup walkthrough (verbatim)

*Reproduced exactly as written in conversation, before any code existed.*

Now, ex1 with sinabs. Plain language, one knob at a time.

**What ex1's neuron does, in words:** it keeps 90% of its charge each step, adds the input in at full strength, fires when it hits 1.0, then empties completely.

That's it. Every setting below just says that in sinabs' language.

---

### The two that actually matter

**1. `tau_mem = 9.4912` — how long it remembers**

You want 90% of the charge left after each step. Sinabs doesn't let you type "90%" — it wants the *memory length* in steps instead, and works out the percentage itself. 9.4912 steps is the memory length that gives you exactly 90%.

**2. `norm_input = false` — don't shrink the input**

Left on (the default), sinabs quietly shrinks every input to 10% before letting it in. Turning it off means the input arrives at full strength. This is the one that saves you the `input_scale: 10.0` workaround Norse needs.

---

### The three that need overriding, or you get the wrong neuron

**3. `spike_fn = SingleSpike` — one spike per step, maximum**

Out of the box sinabs lets a neuron fire *3 or 4 times* in a single step. Your other three frameworks can only ever fire once. Without this override you're not comparing the same thing.

**4. `reset_fn = MembraneReset()` — empty it completely when it fires**

Default behaviour is to only take 1.0 off the top and keep the rest. Ex1 wants the neuron wiped to zero. Same choice you already made as `reset_mechanism: zero` for snnTorch.

**5. `surrogate_grad_fn` — ⚠️ this one is unresolved**

This only affects *learning*, never the neuron's behaviour. Problem: sinabs has no `atan`, which is what your other three use. Same wall you hit with Norse. You'd need to test its four options and pick whichever curve sits closest to `atan(2.0)` — exactly the little side-experiment that landed you on Norse's `circ(0.5)`.

---

### The five you can leave alone

| setting | leave at | plain reason |
|---|---|---|
| `spike_threshold` | `1.0` | already correct |
| `tau_syn` | `null` | off = no extra input smoothing, matches your other three |
| `min_v_mem` | `null` | off = no floor on how negative charge can go, matches your other three |
| `train_alphas` | `false` | stops training from changing the memory length |
| `record_states` | `false` | debug traces only, costs memory |

---

### Two things that block this before any of the above matters

**Sinabs isn't installed.** And when you install it, watch what it does to `torch` — your whole speed comparison assumes one fixed torch version. If `pip install sinabs` moves it, that silently invalidates ex1's existing numbers. Worth installing into a *throwaway* venv first just to look.

**Config alone won't do it.** Ex1 currently knows three frameworks. Adding a fourth needs a new `src/adapters/sinabs_lif.py` plus registering it — the neuron is the only piece that differs, so it's a small file, but it is code, not YAML.

---
---

# Part 2 — what was then built

## 2.1 Files changed

| file | change |
|---|---|
| `src/adapters/sinabs_lif.py` | **new** — the adapter (`build_surrogate`, `build_lif`, `SinabsLIF`) |
| `src/adapters/__init__.py` | `sinabs` added to `FRAMEWORKS` and `IMPLEMENTED`; lazy import wrapped so a missing package gives a readable message |
| `src/config.py` | **new helper** `require_optional_float()` — needed for `tau_syn` and `min_v_mem`, whose "off" value is `null` rather than a number |
| `config/default.yaml` | **new `neuron.sinabs` block**, in the same annotated style as the other three |
| `requirements.txt` | `sinabs==3.1.3` + pinned `nir==1.0.4`, `nirtorch==2.0.2` (see §3.1) |
| `check_env.py` | `sinabs` added to `PACKAGES`, so the appendix records its version |
| `src/plots/style.py` | `sinabs` identity: reddish-purple, diamond marker, dash-dot |

The dataset, architecture, training block and metrics are untouched and shared,
which is the point.

**On the plot identity being safe to add:** `Results.conditions` in
`src/plots/data.py` filters `FRAMEWORK_ORDER` by what is actually PRESENT in the
data, so ex1's and ex2's existing figures — drawn from runs containing no sinabs
rows — are unchanged. `reddish_purple` was also removed from `SPARE_COLOURS` to
preserve the rule that a framework colour is never reused. Verified: four
frameworks, four distinct colours, no overlap with the spares.

## 2.2 The config block, as committed

```yaml
  sinabs:
    tau_mem: 9.4912          # -1/ln(0.9)  -> decay 0.9
    tau_syn: null            # first-order neuron, like the other three
    norm_input: false        # gain 1.0, decoupled -- no input_scale needed
    spike_threshold: 1.0
    spike_fn: single         # override: sinabs defaults to MultiSpike
    reset_mechanism: zero    # override: sinabs defaults to soft (subtract)
    v_reset: 0.0
    min_v_mem: null          # no lower clamp, like the other three
    train_alphas: false
    record_states: false
    surrogate:
      type: single_exponential   # OPEN: sinabs' own default, NOT a matched value
      grad_width: 0.5
      grad_scale: 1.0
```

## 2.3 Verified: the arithmetic lands on the ex1 target

Run against the committed config, with sinabs absent (config parsing and the
decay/gain arithmetic need no package):

```
tau_mem               9.4912
norm_input            False

derived decay      0.900000   (ex1 target 0.9)
derived gain       1.000000   (ex1 target 1.0)
derived DC gain    9.999978   (ex1 target 10.0)
  -> matches the ex1 target neuron
```

The DC gain reads 9.999978 rather than 10.0 because `tau_mem` is written to four
decimal places. Exact would be `9.49122023...`; the shortfall is 2.2e-6, roughly
five orders of magnitude below the 1e-4 float32 yardstick in
`equivalence.tolerance.max_v_deviation`. Not worth more digits, but worth knowing
it is a rounding artefact and not a translation error.

Also confirmed in the same run: snnTorch, SpikingJelly and Norse all still build,
forward, and reset — the additions broke nothing.

## 2.4 Three sinabs behaviours the adapter has to defend against

All three are **silent** — they produce wrong numbers rather than errors — which
is why each is handled explicitly and commented in the adapter.

**1. Dimension 1 is TIME.** `LIF.forward` opens with

```python
batch_size, time_steps, *trailing_dim = input_data.shape
```

The shared network hands over one timestep shaped `(batch, C, H, W)`. sinabs
would read `C` as the number of timesteps and `(H, W)` as the neuron shape, and
raise nothing at all. The adapter's `forward` adds a length-1 time axis and takes
it straight back off:

```python
spikes = self.lif(x.unsqueeze(1)).squeeze(1)
```

**2. `reset_states()` keeps the buffer shape**, so it is not used. Its body is
`buffer.zero_()` — values cleared, shape retained, layer still "initialised" for
one specific batch size. Two consequences:

- A short final batch then takes sinabs'
  `handle_state_batch_size_mismatch()`, which rebuilds the state with
  `torch.randint` — it fills each neuron's membrane by copying a **random other
  sample's**. That is harmless only while the buffer happens to be all zeros, and
  depending on that is not acceptable.
- `is_state_initialised()` would keep returning `True` after a reset, so
  `has_state()` could not honestly report whether the reset worked — which is
  precisely what `check_network.py` exists to verify.

So `reset()` restores the zero-size tensor `register_buffer` originally held. The
next forward re-infers the shape from its real input, giving the same "fresh
neuron" semantics as the snnTorch and Norse adapters. This is the same kind of
deliberate divergence from a library's own reset call as the snnTorch adapter's
refusal to use `utils.reset`.

**2b. `tau_mem` is a trainable `nn.Parameter`, and `train_alphas: false` does not
change that.** Found by the **first `check_network.py` run**, from the parameter
count — not by reading the source:

```
snntorch        18,254
spikingjelly    18,254
norse           18,254
sinabs          18,257   <-- three extra, one per LIF layer
```

The architecture accounts for 18,254 exactly (`612 + 9,632 + 8,010`). The three
extras are one `tau_mem` per LIF layer. `train_alphas` chooses *which* quantity is
the parameter (τ when false, α when true); it does not decide *whether one exists*.

Two consequences, had it gone unnoticed: sinabs would have been the only framework
of the four whose decay drifted away from the configured 0.9 during training, and
the "identical network" claim would have been false by three parameters — with
`report_ex1.md` reporting a parameter count that silently differed for one
framework.

`SinabsLIF.__init__` now calls `requires_grad_(False)` on the neuron's parameters.
This is unconditional, not a config flag, on the same reasoning by which
`build_lif_node()` refuses SpikingJelly's `step_mode='m'` outright: the other three
frameworks hold their time constants as plain constants and none of them *can*
learn one here, so a configurable freeze would only offer the choice of making
sinabs incomparable. Learning time constants would be a separate experiment that
all four frameworks take part in. `describe()` records `tau_mem_trainable: false`
so the intervention lands in the results file.

**3. The defaults aim at ANN→SNN conversion, not at this comparison.**
`MultiSpike` + `MembraneSubtract` + no leak are exactly the three conditions
under which a spiking neuron's firing rate equals ReLU — the guarantee sinabs
exists to provide. ex1 overrides the first two. This is worth stating in the
write-up as a *difference in purpose*, not as sinabs being misconfigured. Full
derivation: `local_docs/SNNs_Introduction_BaseConcepts.md` §4.

---

# Part 3 — how to run it

## 3.1 Install — and the torch worry, now settled

The concern raised in Part 1 — that installing sinabs might move `torch` and
invalidate the runs already recorded — was **checked and is unfounded.** Measured
with `pip install --dry-run --report` against this project's venv:

```
Would install/change:
  sinabs      3.1.3
  nir         1.0.4
  nirtorch    2.0.2
  samna       0.48.6
```

Four packages, and **that is the complete list.** `torch`, `numpy`, `h5py`, `pbr`
and `matplotlib` are all absent from it, meaning pip is satisfied with the
versions already present. Two things follow:

- **torch is not moved.** sinabs declares only `torch>=1.8`, which the existing
  install already satisfies. The recorded ex1 timings stay valid, so the existing
  three runs do **not** need redoing.
- **numpy stays below 2.0.** sinabs declares `numpy` unpinned and samna declares
  no dependencies at all, so nothing pushes against the `numpy<2.0` bound that
  tonic requires.

All four are pure-python `py3-none-any` wheels, so there is no platform-specific
build to fail on Colab or Kaggle.

**The one real snag, already pinned around.** sinabs declares `nir<=1.0.4` while
the current `nirtorch` (2.6) declares `nir>=1.0.6` — mutually exclusive. pip
resolves it by backtracking to `nirtorch 2.0.2`, which is slow and, worse, could
silently resolve differently once either package publishes again. Hence the pins
in `requirements.txt`. Note also that `nir` is **not** optional despite being
unused here: `sinabs/__init__.py` runs `from .nir import from_nir, to_nir` at
import time, so `pip install sinabs --no-deps` would produce a sinabs that cannot
be imported.

So the install is just the normal one:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```

and on Colab/Kaggle, as before, `pip install -r requirements.txt`.

**Note:** until you do this locally, `check_env.py` will exit 1 and report
`sinabs  NOT INSTALLED`. That is correct — the environment genuinely no longer
matches `requirements.txt` — not a new fault.

## 3.2 Then verify before measuring

```powershell
.venv\Scripts\python check_network.py --config config/default.yaml --framework sinabs --experiment ex1
```

This is the run that confirms the three defences in §2.4 actually hold. Expect to
check three things in its output:

1. **spike shapes per layer** match the other three frameworks — this is the
   `unsqueeze(1)` working. If the layer shapes look wrong, defence 1 failed.
2. **`has_state()` is False after a reset** — defence 2 working.
3. **spike rate is a fraction below 1.0** — `spike_fn: single` working. Anything
   above 1.0 means MultiSpike is still active.

## 3.3 Only then, the measured run

```powershell
.venv\Scripts\python train.py --config config/default.yaml --framework sinabs --experiment ex1
```

Same command shape as the other three frameworks — `--config` is required, and
`--experiment ex1` decides the output folder. For a fast smoke test first, add
`--epochs 1 --max-batches 5`.

---

# Part 4 — open items, carried forward

These are genuinely unresolved. None of them blocks a run, but each one limits
what the numbers may be claimed to mean.

1. **The surrogate is not matched.** `single_exponential(0.5, 1.0)` is sinabs' own
   default, chosen so the config is runnable and honest. The other three
   frameworks all use a curve tuned against `atan(alpha=2.0)`. Until the same
   curve-fitting is done for sinabs — the exercise that produced Norse's
   `circ(0.5)` — sinabs' gradient scale relative to the other three is **unknown**,
   and any accuracy difference cannot be attributed to the framework rather than
   to the surrogate. This is the single biggest caveat.
2. **The `tau_mem` rounding is visible in the equivalence check.** Not a defect,
   but it must not be mistaken for one. `tau_mem: 9.4912` gives
   `decay = 0.899999784`, an error of −2.16e-07 per step against the exact 0.9,
   which accumulates to the ~1e-06 deviations in §5. The exact value is
   `tau_mem = 9.4912215810`. Writing more digits would drop sinabs to float32
   epsilon like Norse; leaving it is also defensible, since 1e-06 is 100× under
   the 1e-04 reference. **Your call** — it is a config value.
3. **The adapter was written against the `develop` branch, but `requirements.txt`
   pins the 3.1.3 release.** Every quoted source line came from `develop`. The two
   should be the same for these classes, but that is an assumption, not a check —
   and it is the most likely cause if the first run fails on a signature mismatch.
4. **Nothing has been executed.** Every claim about sinabs' behaviour in this
   document is read from source, not observed. The first run is a verification run.

---

# Part 5 — equivalence check: measured, four-way

`equivalence_check.py` was extended to four frameworks and **run**. It now skips
any framework it cannot import, prints why, and records both
`frameworks_compared` and `frameworks_skipped` in the summary JSON — so a
three-framework summary can never later be mistaken for a four-way run in which
the fourth silently agreed.

## 5.1 Result: sinabs matches the other three

| pair | max\|dv\| pre | spike times |
|---|---|---|
| snntorch vs spikingjelly | 0.000e+00 | 100 % |
| snntorch vs norse | 1.192e-07 | 100 % |
| **snntorch vs sinabs** | **8.345e-07** | **100 %** |
| **norse vs sinabs** | **1.192e-06** | **100 %** |

Worst across both input tests: **1.192e-06**, against the 1.0e-04 reference —
100× inside it. Spike times match **100 %** for every pair, on both the
`constant_step` and `poisson` inputs. All four fire 4 spikes on poisson, 8 on
constant_step, with identical first-spike steps.

**The translation is correct.** sinabs' deviation is larger than Norse's for one
reason only, and it is arithmetic rather than a mismatch — see open item 2.

## 5.2 A genuine new finding: sinabs uses `v >= threshold`

The boundary probe now splits **2 against 2**:

```
snntorch       v = 1.0    no spike  -> rule is  v >  threshold
spikingjelly   v = 1.0    FIRES     -> rule is  v >= threshold
norse          v = 1.0    no spike  -> rule is  v >  threshold
sinabs         v = 1.0    FIRES     -> rule is  v >= threshold
```

This follows from `SingleSpike`'s `(v_mem - spike_threshold >= 0)`. It is
hard-coded and cannot be configured away, and it only bites when the membrane
lands exactly on the threshold — which is why the equivalence amplitudes stay
below 1.0. Previously this was a 1-against-2 split; it is now 2-against-2, which
is a better sentence for the write-up than "SpikingJelly is the odd one out".

---

**Closed since Part 1 was written:**

- ~~`equivalence_check.py` hardcodes three frameworks.~~ Extended, run, and
  reported above.

- ~~Whether ex1's existing three runs need redoing.~~ **No.** torch is not moved
  (§3.1, measured), so the recorded timings stay valid.
- ~~`src/plots/style.py` has no sinabs identity.~~ Added, and verified not to
  disturb the existing figures (§2.1).

---

## Related

- `local_docs/Intro-to-Sinabs.md` — full sinabs briefing, parameter tables, source quotes
- `local_docs/SNNs_Introduction_BaseConcepts.md` — α, τ, DC gain, ReLU, IAF vs LIF
- `local_docs/norse_superspike_alpha_finding.md` — the precedent for open item 1
- `experiments/ex1/report_ex1.md` — ex1's results and write-up
- `experiments/ex2/ex2_design.md` — the design-doc format this follows
- `config/default.yaml` — the ex1 target neuron, and all four translations
