"""Shared test configuration.

The suite must run on a CPU-only box with no CAMELS data, no GPU, no network
and no trained checkpoint, so every test builds its own synthetic arrays.

Torch is pinned to a single thread: the UNets exercised here are deliberately
tiny, so thread pools cost far more than they save, and a single thread keeps
the suite from oversubscribing a shared/CI machine.
"""

import torch

torch.set_num_threads(1)
torch.use_deterministic_algorithms(False)
