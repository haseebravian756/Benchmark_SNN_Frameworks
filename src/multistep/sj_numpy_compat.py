"""Fix for a stale numpy alias that breaks SpikingJelly's CUDA kernels.

SpikingJelly 0.0.0.0.14 reads `np.int` at `auto_cuda/base.py:249`. Numpy REMOVED
that alias in 1.24, so every cupy kernel launch raises

    AttributeError: module 'numpy' has no attribute 'int'

**Pinning numpy cannot fix this.** Colab runs Python 3.12, and the earliest numpy
with 3.12 wheels is 1.26 -- there is no installable numpy on which the line works.
So the library has to be corrected at runtime.

Two things make that safe:

  * The line sits in `check_ctypes`, a pure VALIDATION method: it asserts that an
    integer cupy array was declared with an `int` ctype, and computes nothing. It
    is nevertheless called on every kernel launch (`base.py:326`) with no flag to
    disable it, so it cannot simply be skipped.

  * `np.int` was literally an alias for the builtin `int`. Numpy's own deprecation
    message states that substituting `int` "will not modify any behavior and is
    safe".

The replacement below reproduces the original method exactly, with that one
substitution. It asserts no less than the original did.

WHY HERE AND NOT IN site-packages: a `site-packages` edit vanishes on every Colab
reconnect and is invisible to `pip freeze`. Keeping the fix in the repo means it is
versioned, applied automatically, and visible to anyone reading the results.

Verified: with this applied, `'m'`+cupy produces spikes BIT-IDENTICAL to the
current pipeline and gradients agreeing to 1.19e-07. See
local_docs/spikingjelly_multistep_intro.md sections 1.3 and 6.3.

This patch must be DECLARED AS A LIMITATION in any report using these numbers.
"""

from __future__ import annotations

import torch

# Set once so repeated builds do not re-announce. Read by `describe()` in the
# adapter, so the results file can record that the patch was in force.
_applied = False

DESCRIPTION = (
    "spikingjelly.activation_based.auto_cuda.base.CKernel.check_ctypes "
    "(np.int -> int; the alias was removed in numpy 1.24)"
)


def is_applied() -> bool:
    return _applied


def apply() -> bool:
    """Install the corrected `check_ctypes`. Returns True if this call did it.

    Idempotent: safe to call from every network build.
    """
    global _applied
    if _applied:
        return False

    import numpy as np

    from spikingjelly.activation_based.auto_cuda import base as ac_base

    def check_ctypes(self, py_dict: dict) -> None:
        for key, value in py_dict.items():
            ctype: str = self.cparams[key]
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.float:
                    assert ac_base.startswiths(ctype, ("const float", "float"))
                elif value.dtype == torch.half:
                    assert ac_base.startswiths(ctype, ("const half2", "half2"))

            if ac_base.cupy is not None and isinstance(value, ac_base.cupy.ndarray):
                if value.dtype == np.float32:
                    assert ac_base.startswiths(ctype, ("const float", "float"))
                elif value.dtype == np.float16:
                    assert ac_base.startswiths(ctype, ("const half2", "half2"))
                elif value.dtype == int:  # was `np.int`, removed in numpy 1.24
                    assert ac_base.startswiths(ctype, ("const int", "int"))

    ac_base.CKernel.check_ctypes = check_ctypes
    _applied = True
    return True
