# src/behav_modeling/config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class RecoveryConfig:
    shift_types: List[int] = field(default_factory=lambda: [6, 8])
    models: List[str] = field(
        default_factory=lambda: ["p", "pv", "pf", "pvi", "pif", "pvf"]
    )
    rbfs: int = 30
    lambda_reg: float = 3
    n_trial_samples: int = 10
    n_sim_runs: int = 30
    gp_scalars: List[int] = field(default_factory=lambda: [1, 3, 5])
    n_jobs: int = 4
