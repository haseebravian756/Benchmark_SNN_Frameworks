# Excluded runs

Runs removed from the CSVs, kept here as evidence.

## 20260801_235851_snntorch_seed1

**Reason: contaminated timing.** The Colab session had restarted and
`prepare_data.py` was not re-run, so this run -- the first of that session --
built the whole 60,000-sample Tonic cache while training.

- epoch 1: **953.0 s**; epochs 2-5: 135.1 / 128.9 / 126.4 / 124.2 s
- recorded `train_time_s` 1467.6 s against ~624 s for the clean rerun
- `train_energy_j` 63.3 kJ, roughly double
- dynamic energy was barely affected (9.6 vs 10.7 kJ) because the GPU sat
  idle during cache building and idle-subtraction removes that

Accuracy, latency, memory and spike rate were unaffected, but the run was
dropped so that per-seed aggregates are not double-counted.

Replaced by `20260802_122610_snntorch_seed1` (warm cache, epoch 1 = 133.0 s).
