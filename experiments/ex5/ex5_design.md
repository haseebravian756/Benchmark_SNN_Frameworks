# Experiment 5 — Design: the input pipeline with denoising switched off

**Status: DESIGN STUB. Config committed, nothing run yet. Written 2026-08-23.**

> **The one-sentence version.** Every run in ex1–ex4 fed the network events that had
> already been filtered by `tonic.transforms.Denoise(filter_time=10000)`; ex5 removes
> that one transform and nothing else, to measure what the filter was actually worth.

Claims marked **[installed]** were verified by reading the installed `tonic`, not from
a web page. Claims marked **[measured]** come from a run on this project's own
hardware, with the command that produced them named. **Nothing in this document is
[measured] yet** — that is the work this stub describes.

---

## 1. Where this experiment sits

ex1 and ex2 vary the **neuron**. ex4 varies the **execution strategy**. ex5 is the
first experiment to vary the **data** — and it is the only one whose variable sits
upstream of the framework entirely.

That has a useful consequence: because the change happens before any framework code
runs, all four frameworks receive byte-identical input either way. So ex5 does not
need four frameworks to answer its question (§5.2).

### 1.1 Why it is worth asking at all

Denoising was inherited, not chosen. The value `10000` comes from the snnTorch/Tonic
N-MNIST tutorial this architecture is based on, and `config/default.yaml:36-43` says as
much. An inherited default that has never been tested is exactly the kind of thing a
developer decision guide should test — a reader deciding whether to denoise their own
event data currently has no number from us to go on.

---

## 2. The question, stated precisely

> Holding the neuron, the network, the framing, the optimiser, the seed and the machine
> fixed, **what does removing the spatio-temporal denoise filter cost or save** — in
> accuracy, in spike rate, and in the energy proxy?

Three separate measurements, and they are expected to move in *different directions*:

- **Accuracy** — unknown sign. Noise events may hurt (they are distractors) or help
  (they are regularisation). This is the actual open question.
- **Spike rate** — expected UP. More events survive to reach `ToFrame`, so frame values
  are larger and more pixels are non-zero.
- **Energy proxy** — expected UP, because it is derived from spike counts. Per the
  project's standing rule, this gets **reported, not interpreted**.

### 2.1 What ex5 does NOT ask

- It does not ask what the *best* `filter_time` is. That is a sweep, not an ablation —
  ex5 has two points, `10000` and off.
- It does not ask whether denoising helps on any other dataset. N-MNIST only.
- It does not ask about a *real* event camera's noise. N-MNIST was recorded by moving a
  DVS in front of a static MNIST image on a screen, so its noise is that sensor's, at
  that temperature, at those bias settings. §5.3.

---

## 3. What changes, and what deliberately does not

**Changed — one key:**

```yaml
dataset:
  denoise_filter_time_us: null
```

**[installed]** `null` does not mean "call tonic with denoising disabled" — tonic has no
such setting. It means the transform is never constructed. `src/data.py:341-348`:

```python
stages: list[Callable] = []
if identity.denoise_filter_time_us is not None:      # <- skipped when null
    stages.append(tonic.transforms.Denoise(filter_time=...))
stages.append(identity.to_frame)
stages.append(PadOrCropFrames(identity.time_steps))
```

So the pipeline goes from `Denoise → ToFrame → PadOrCrop` to `ToFrame → PadOrCrop`.

**Unchanged, and this is the whole basis of the comparison:** framing (T=20,
`n_time_bins`), `binarize: false`, `first_saccade_only: false`, `stabilize: false`, the
neuron, the architecture, the optimiser, the learning rate, the loss, the batch size,
the seeds, and the metrics. All inherited from `default.yaml` via `extends`.

### 3.1 A separate cache, not an overwritten one

Denoise runs before framing and changes which events exist, so it is part of the cache
identity (`src/data.py:189-190`). ex5 therefore builds `cache/nmnist/t20_dn0_<digest>/`.
The ex1 cache `t20_dn10000_7e3016` is neither touched nor invalidated — both live on
disk, and switching between them is switching config. Nothing to clean up, nothing to
rebuild afterwards.

---

## 4. What the filter being removed actually does

Worth stating precisely, because the report will need it and because the mechanism is
what licenses the interpretation.

**[installed]** `tonic/functional/denoise.py`, in full logic:

```python
timestamp_memory = np.zeros((width, height)) + filter_time
for event in events:
    x, y, t = int(event["x"]), int(event["y"]), event["t"]
    timestamp_memory[x, y] = t + filter_time          # (1) this pixel stays "live" for filter_time
    if (  (x > 0            and timestamp_memory[x - 1, y] > t)     # (2) is any of the 4
       or (x < width - 1    and timestamp_memory[x + 1, y] > t)     #     neighbours still live?
       or (y > 0            and timestamp_memory[x, y - 1] > t)
       or (y < height - 1   and timestamp_memory[x, y + 1] > t)):
        events_copy[copy_index] = event                # (3) if yes, keep
        copy_index += 1
```

An event survives only if one of its **4** immediate neighbours fired within the last
`filter_time` µs. Two properties follow, and both matter:

- **It is a correlation test, not a noise model.** It has no idea what noise looks
  like; it deletes whatever is spatially isolated in time. DVS background activity
  (thermal noise and junction leakage, firing at random uncorrelated pixels) fails that
  test; a moving edge, which lights a contiguous line of pixels within microseconds,
  passes it.
- **It is causal and streaming** — one timestamp per pixel, no lookahead. For DVS128
  that is 128×128×4 B ≈ 64 KB. This is why the same filter ships in real hardware
  pipelines (iniVation DV's `BackgroundActivityFilter`, Prophesee Metavision's
  activity/STC filters), and therefore why "denoise on" is a deployable choice rather
  than a training-time convenience.

**TO MEASURE:** how many events the filter actually drops on N-MNIST. This is the
single most useful number in the experiment and it is not yet known. If it drops 2% of
events, a large accuracy effect would be surprising and worth double-checking; if it
drops 40%, the opposite. Get this before interpreting anything else.

---

## 5. What you may and may not claim

**TO FILL after the runs.** The traps that are already visible:

### 5.1 Train/inference preprocessing must match — do not present this as a robustness result

ex5 trains *and* evaluates without denoising. That is a clean ablation of the pipeline.
It is **not** a test of whether an ex1-trained model survives noisy input at inference
— that would be train-denoised / test-raw, a different experiment, and a mismatch of
that kind is a distribution shift that would confound the result.

- ✅ "A pipeline without denoising reaches X% where the denoised pipeline reaches Y%."
- ❌ "The model is robust to sensor noise." Not shown. Nothing was tested under
  mismatch.

### 5.2 One framework is sufficient — say why, do not apologise for it

The variable is upstream of every framework, so all four consume identical frames. Runs
on additional frameworks would be replication, not coverage. If only one framework is
run, state that reason explicitly in the report; it is a design property, not a gap.

### 5.3 N-MNIST's noise is one sensor's noise

The noise being filtered is the ATIS sensor's, at the temperature and bias settings of
that 2015 recording session. Real noise rates vary with illumination, temperature and
bias. So ex5 bounds nothing about other cameras.

### 5.4 If the effect is small, check it against the noise floor first

ex1 established a run-to-run spread across seeds. An accuracy difference inside that
spread is not a result. Reuse the ex1 noise-floor argument rather than inventing a new
one.

---

## 6. Concrete plan

```
# ── 1. build the cache. Slow, once, both splits. Separate directory from ex1 (§3.1).
.venv\Scripts\python prepare_data.py --config config/config_no_denoise.yaml \
    --experiment ex5

# ── 2. confirm the network is what you think it is before spending GPU time.
.venv\Scripts\python check_network.py --config config/config_no_denoise.yaml \
    --framework snntorch

# ── 3. the runs. Same seeds as ex1 so the comparison is paired (§5.4).
python train.py --config config/config_no_denoise.yaml --experiment ex5 \
    --framework snntorch --seed 0 --results-root <drive>
#   ... repeat for seeds 1, 2

# ── 4. baseline for comparison: ex1's existing runs. Nothing to re-run.
#       experiments/ex1/results/runs/ already holds 4 frameworks x 3 seeds.
```

### 6.1 Scope still to decide

One framework × 3 seeds, or all four × 3 seeds. §5.2 argues one is sufficient; four
costs 4× the Colab time and buys replication. **Owner's call, not settled here.**

---

## 7. Open item this surfaces for the merge

`comparison_pipeline/SNNs-auf-GPUs/event_data_workflow/data_pipeline.py:297` hardcodes
the filter time:

```python
frame_tf = ComposedTransform([transforms.Denoise(filter_time=10000), ...])
```

So ex5 is runnable through *this* project's config path only. Making it runnable in the
merged pipeline means promoting that literal to a parameter — a small change, but one
that belongs in `docs/merge_acceptance_gates.md` rather than being done silently here.

---

## 8. Sources

- `tonic/functional/denoise.py`, `tonic/transforms.py:101-119` — installed, read
  2026-08-23.
- `config/default.yaml:36-43` — where the inherited `10000` is documented.
- `src/data.py:189-190`, `:290`, `:341-348` — the off-switch and the cache identity.
- N-MNIST: Orchard et al. (2015), *Converting Static Image Datasets to Spiking
  Neuromorphic Datasets Using Saccades*.

## Related

- `experiments/ex1/report_ex1.md` — the baseline ex5 is measured against.
- `docs/open_items.md` — where this should be registered.
