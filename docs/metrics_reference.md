# Metric reference

One table, for looking up what a number means. Design rationale, execution order
and CSV schema live in `docs/metrics_plan.md`; deferred ideas in
`docs/open_items.md`.

---

| metric | unit | in plain words | how it is measured | what it means for N-MNIST |
|---|---|---|---|---|
| **test accuracy** | % | How often the network picks the right answer on data it has never seen. | `100 × correct / total`. The prediction is the output neuron that fired most across all T timesteps: `spike_counts.argmax(dim=1)`. | Of the 10,000 held-out event recordings of handwritten digits, the share it labels with the correct digit 0–9. ~98–99% is the published ceiling for this architecture. |
| **training time** | s | How long the whole learning process took, start to finish. | `time.perf_counter()` around the training loop, with `torch.cuda.synchronize()` on both sides so GPU work is finished, not just queued. Warm-up runs first and is excluded. | Wall-clock to make `epochs` full passes over 60,000 recordings, each 20 frames of 2×34×34. Includes data loading, which is identical for all three frameworks. |
| **inference throughput** | samples/s | How many samples the network can chew through per second when they arrive in bulk. Higher is better. | `n_samples / elapsed_s` over the whole batched test pass at `batch_size`. MLPerf's *Offline* scenario. | How many digit recordings get classified per second when fed in batches of 128 — the number that matters for processing a stored dataset. |
| **inference latency** | ms | How long **one** sample takes, from feeding it in to the answer coming out. Lower is better. Not the inverse of throughput. | `t_output_ready − t_input_fed` at `batch_size = 1`, with `torch.cuda.synchronize()` on both sides. Reported as median over N samples, plus mean and p90. MLPerf's *Single-Stream* scenario. | Time from handing the network one 20-frame recording to getting its digit back — the number that matters for a live event camera, where samples arrive one at a time. |
| **spike rate** | % | How busy the neurons are: the share of chances to fire that a neuron actually took. Lower means the same job done with less activity. | `100 × spikes / (neurons × T × samples)`, counted in a dedicated pass with counting switched on, because counting itself costs GPU time. Per LIF layer plus a spike-weighted overall figure. | Across 20 timesteps, how often the average neuron fires. 12% means it fires on roughly 2.4 of the 20 frames. This is the efficiency argument for SNNs — and it is what maps to power draw on real neuromorphic hardware. |
| **peak memory (allocated)** | MB | The most GPU memory the model's tensors ever needed at once. | `torch.cuda.max_memory_allocated()`, after `reset_peak_memory_stats()` at the start of the region. | Memory to hold a batch of 128 recordings × 20 timesteps, plus every intermediate activation kept for backpropagation-through-time. Measured 0.88 GB on a T4. |
| **peak memory (reserved)** | MB | How much memory PyTorch actually took from the driver — usually more than it handed to tensors. | `torch.cuda.max_memory_reserved()`, same region. | Closer to what `nvidia-smi` shows. A framework whose allocation pattern fragments memory reserves noticeably more than it allocates, and that gap is itself a framework property. |
| **training energy** | J | Electrical energy the GPU consumed while learning. | GPU power polled in a background thread via `nvmlDeviceGetPowerUsage` and integrated over time: `E = Σ (Pᵢ+Pᵢ₊₁)/2 × Δtᵢ`. Whole training run only. | Joules spent learning to recognise digits from 60,000 event recordings. Comparable **between frameworks measured identically on one machine** — not an absolute physical figure (see the caveat below). |
| **training energy (dynamic)** | J | The same, minus what the GPU would have burned sitting idle for that long. | `E_total − P_idle × duration`. Idle power is measured immediately after training, when the GPU is idle but still at working temperature. | The energy attributable to the training itself rather than to the machine being switched on. The idle baseline is the only assumption in the energy numbers, which is why the unsubtracted total is reported next to it. |

---

## Two things to state whenever these numbers are quoted

**Throughput and latency are not inverses.** Throughput is measured with 128
samples in flight at once; latency with one. A system can be excellent at one and
poor at the other. `1000 / throughput` yields "average ms per sample under
batching" — a third, different number.

**NVML energy is relative, not absolute.** The sensor refreshes at only ~10 Hz,
reports a time-averaged rather than instantaneous value, and published
comparisons against physical power meters show errors up to 73%. So these figures
support *"framework A used more energy than framework B on this machine, measured
identically"*. They do not support *"this model costs X joules"*. This is a
stronger caveat than the reference doc's "energy is a proxy for framework
overhead", and it should replace that wording.
