from dataclasses import dataclass, field
from typing import List


@dataclass
class ClusterConfig:
    regions: List[str] = field(default_factory=lambda: ["hpc", "acc"])
