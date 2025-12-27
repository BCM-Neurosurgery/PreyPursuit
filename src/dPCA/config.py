# src/behav_modeling/config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class DPCAConfig:
    window_left: int = 14
    window_right: int = 16
    reg_strength: float = 0.5
    bias: float = 0.05
    regions: List[str] = field(default_factory=lambda: ["hpc", "acc"])
