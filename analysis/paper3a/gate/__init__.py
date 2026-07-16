"""WP-A3 gate compute: run the WP-A2 operators over every painted design point.

Produced by the 2026-07-16 rusty session as A3 prep. `process_run_snapshot`
is the unit of Slurm-array work; the A3 session aggregates the per-run npz
tables into the envelope/overlay figures and the decision memo.
"""

from .gate_operators import GateConfig, process_run_snapshot

__all__ = ["GateConfig", "process_run_snapshot"]
