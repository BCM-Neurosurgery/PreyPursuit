from dataclasses import dataclass

@dataclass
class BGLMConfig:
    shift_type: int = 6
    model: str = 'p'
    rbfs: int = 30
    lambda_reg: float = 3