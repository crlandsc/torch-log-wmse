"""Shared construction helpers for the test suite.

Every test constructs the metric through the factories here rather than calling `LogWMSE(...)`
directly. The point is blast radius: the 1.0.0 redesign changes the constructor twice - once when
`audio_length` disappears, once when the class splits into `LogWMSE` / `LogWMSELoss` - and routing
every construction through one function means each of those changes edits one place instead of
forty call sites.

`tests/` is a package (it has `__init__.py`), so pytest puts the REPO ROOT on sys.path and this
module is `tests.conftest` - import it that way, not as bare `conftest`.

Two things this file must keep doing:

* **Insert the repo root at sys.path[0], not append.** With append, a pip-installed copy of the
  package in site-packages shadows the working tree and the suite silently tests the installed
  wheel instead of the code under edit. pytest imports conftest.py before any test module, so
  doing it here makes the working tree authoritative for the whole suite.
* **Import nothing outside torch and the package.** No numpy, no matplotlib. The package needs
  neither, so a test-collection dependency on them means the suite cannot run against a bare
  install of what it is testing - which is exactly how a matplotlib import once broke CI while
  passing locally.
"""
import math
import os
import sys

# See the module docstring: insert(0, ...), never append.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from torch_log_wmse import LogWMSE
from torch_log_wmse.constants import EPS, SCALER
from torch_log_wmse.freq_weighting_filter import HumanHearingSensitivityFilter

# Polite on a shared machine, and a fixed thread count is also what keeps golden values
# reproducible - FFT reductions are not bit-identical across thread counts.
torch.set_num_threads(2)

SR = 44100
CEILING = float(SCALER * math.log(EPS))  # +73.6827..., the value for an exact match


def make_metric(audio_length=1.0, sample_rate=SR, **kw):
    """The POSITIVE metric: higher is better.

    `return_as_loss` is accepted and forwarded so that call sites which deliberately exercise both
    signs keep working. When the class splits, this function chooses the class and the keyword
    disappears from the public API - callers do not change.
    """
    kw.setdefault("return_as_loss", False)
    return LogWMSE(audio_length=audio_length, sample_rate=sample_rate, **kw)


def make_loss(audio_length=1.0, sample_rate=SR, **kw):
    """The LOSS: the negated metric, for training."""
    kw["return_as_loss"] = True
    return LogWMSE(audio_length=audio_length, sample_rate=sample_rate, **kw)


def make_filter(audio_length=1.0, sample_rate=SR, **kw):
    """The frequency-weighting filter on its own.

    `audio_length` is what pins the transform size today; it is derived from the input later, so
    this argument becomes inert rather than disappearing from call sites.
    """
    return HumanHearingSensitivityFilter(audio_length=audio_length, sample_rate=sample_rate, **kw)


def bipolar(*shape, seed=None, scale=1.0):
    """U(-1, 1) * scale, seeded on request.

    torch's generator, never numpy's: numpy is not a dependency of the package, and a golden value
    that depends on numpy's RNG cannot be reproduced in a bare environment.
    """
    if seed is not None:
        torch.manual_seed(seed)
    return (2 * torch.rand(*shape) - 1) * scale
