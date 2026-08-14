# Open items

Things deliberately deferred, one line each. Not a backlog of bugs — these are
choices made to postpone, kept here so they are not silently forgotten.

> **The experiments below are IDEAS, not a committed plan.** Only Experiment 1
> is real. Nothing here should be treated as decided, scheduled, or something to
> build towards.

Add to it as things come up. Move an item out when it is done or abandoned.

---

## Experiments

| # | item | status |
|---|---|---|
| E1 | N-MNIST, matched neuron parameters | **DONE** — 3 seeds, 9 runs, `experiments/ex1/report_ex1.md` |
| E2 | N-MNIST with each framework's own out-of-the-box parameters. **Designed** — `experiments/ex2/ex2_design.md`, `config/config_ex2.yaml` written and validated (no code changes needed). Equivalence check is expected to FAIL and that failure is the result. | **ready to run** |
| E3 | Second dataset (DVS128 Gesture is the intended next one) — needs a `DATASETS` entry and probably different architecture knobs | not started |
| E4 | Other neuron types each framework offers | not started |
| E5 | **SpikingJelly in its FASTEST mode — multi-step (`step_mode='m'`) + `cupy` backend.** This is the mode its own paper claims 11x training acceleration for, and every ex1/ex2 run measured SpikingJelly *without* it, so those speed figures are a **lower bound**. **BUILT** — `src/multistep/`, `config/config_ex4_multistep.yaml` (fused kernel) and `config/config_ex4_control.yaml` (multi-step *without* the kernel, the control that attributes the speedup). It IS a config flag now: `step_mode: m` routes the build, and every `'s'` config takes the untouched original path. Two corrections to what this row used to say: the conv layers do **not** need `SeqToANNContainer` — folding T into the batch dimension is plain torch and verified identical — and no rewrite of `src/network.py` was needed (LIFNode carries zero parameters, so LIF layers are swapped after `build_network` runs). Measured 4.99x on the neuron, spikes bit-identical, gradients to 1.19e-07. Needs `cupy-cuda12x<14` (14+ demands numpy>=2, which breaks tonic) and a runtime fix for a `np.int` call SpikingJelly never updated — **declare both as limitations**. Design, verification and execution flow: `local_docs/spikingjelly_multistep_intro.md`. | **ready to run** |
| E6 | Norse `circ(0.5)` vs `super` — quantifies surrogate choice alone | **ready to run**, `config/norse_super.yaml` exists |
| E7 | `detach_reset` ablation — SpikingJelly only, where the framework already exposes it. No custom work for the other two | ready to run |
| E8 | Multi-seed runs (3 seeds) + mean ± std reporting | **DONE** — now the standard for every experiment |

## Metrics

| # | item | status |
|---|---|---|
| M1 | Energy (training + inference) via NVML | **phase 2**, columns already in the schema |
| M2 | Synaptic operations (SOPs) = `Σ(spikes_k × fanout_k)`, the standard neuromorphic energy proxy — derivable at analysis time from per-layer spike counts + per-layer fan-out, no new measurement | not started |
| M3 | Per-timestep spike rates — does activity decay across T? Hook is already per-layer; would multiply stored data by T | not started |
| M4 | SNN-sense "latency": accuracy vs number of timesteps processed (anytime / early-exit inference). Distinct from wall-clock latency, same word. Changes the evaluation loop | not started |
| M5 | Per-batch timing via CUDA events (low overhead, unlike per-batch `synchronize`) | not started |
| M6 | External power-meter validation of the NVML energy numbers — literature reports NVML error up to 73% vs a physical meter | unlikely to be feasible; note as a limitation |
| M7 | Spike rate DURING training (currently measured on inference only). Would be a moving target since weights change every batch, and would need its own untimed pass. Different question: "how active is the network while learning?" vs "how active is the deployed network?" | not started |
| M8 | Report `runs.csv` energy in kJ/Wh as well as J — a 5-epoch run is ~20 kJ, which reads awkwardly in joules | trivial, analysis-time |
| M9 | **Gradient norm per epoch** — ex1 measured Norse's 6.03× conv1 gradient norm with a throwaway script; it is now a standing measurement. `gradient_probe()` in `src/metrics.py`, one fixed batch (the warm-up batch, reused so no extra permutation is drawn from the shuffler), forward+backward with NO optimizer step, at epoch 0 and after every epoch. Writes a **new** `gradients.csv` rather than new columns in `epochs.csv`, because `append_row` refuses a changed header and that would have locked every existing results folder. Epoch 0 is the useful one for cross-framework work: identical weights everywhere, so any difference there is the surrogate gradient alone. Verified against the pre-change code — byte-identical losses, accuracies and spike rates | **DONE** |

## Engineering

| # | item | status |
|---|---|---|
| G1 | Architecture into YAML instead of Python constants — the original brief wanted this; deferred at the user's request. Kept in one place in `src/network.py` so lifting it out is easy | deferred by choice |
| G2 | Full 70k cache build cost on Colab is still unmeasured; also unclear where the cache should live so it survives a session restart (Drive is persistent but slow) | open |
| G3 | Machine-specific config via `extends:` | **done** — `config/colab.yaml` writes results to Google Drive so they survive a disconnect |
| G6 | Plots from `runs.csv` / `epochs.csv` / `layers.csv` | **DONE** — `make_plots.py` + `src/plots/`, 17 figures in six families, design and sources in `local_docs/plotting_schema.md`. Generic condition axis, so it serves every experiment |
| G4 | `nn.Sequential` instead of `nn.ModuleList` in `SpikingNet` — purely cosmetic, would shorten the forward by two lines | decided against, revisit only if wanted |
| G5 | Learnable neuron parameters (`learn_beta`, `learn_threshold`; SpikingJelly's `ParametricLIFNode`) — all three support it; "does letting the neuron learn help?" is a legitimate follow-up. Would change the parameter count, so all three must do it or none | not started |

## External

| # | item | status |
|---|---|---|
| X1 | Norse: request a release containing the SuperSpike `alpha` fix (already merged as #403, unreleased since v1.1.0). Draft issue text in `docs/norse_superspike_alpha_finding.md` §7 | user's call |
