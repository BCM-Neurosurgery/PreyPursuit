from dataclasses import dataclass


@dataclass
class NeuralConfig:
    fit_type = "vi"
    n_steps = 10_000
    optimizer = "scheduled"
    guide = "normal"
    credible_interval = 95
    n_jobs = 4
