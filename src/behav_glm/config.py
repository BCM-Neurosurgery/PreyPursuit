from dataclasses import dataclass
from typing import Optional


@dataclass
class BGLMConfig:
    null_config: bool = True
    model: Optional[str] = None
