"""SpikingJelly's multi-step mode, isolated from the working pipeline.

Every result in experiments/ex1 and experiments/ex2 was produced with SpikingJelly
in `step_mode='s'`, which cannot reach its fused CUDA kernel. This package lifts
that limitation -- open item E5 -- WITHOUT editing the code that produced those
results, so they stay reproducible from the same commit.

Nothing in `src/network.py`, `src/adapters/` or `src/metrics.py` was changed. The
only edits outside this package are single-line swaps of `build_network` for
`build_net` at the three places a network gets built.

Reachable only via `neuron.spikingjelly.step_mode: m` in a config. Every existing
config says `s`, so for all of them this package is never even imported.

    build.build_net(framework, neuron_cfg, info, seed)   the one decision point
    sequence_net.SequenceSpikingNet                      folds T into the batch
    sj_lif_multistep.SpikingJellyMultiStepLIF            the multi-step LIF
    sj_numpy_compat.apply()                              a required library fix

Measured on a Colab T4: 4.99x on the neuron alone, with spikes BIT-IDENTICAL to the
single-step pipeline and gradients agreeing to 1.19e-07.

Design, the verification that gates it, and the full execution flow:
local_docs/spikingjelly_multistep_intro.md
"""
