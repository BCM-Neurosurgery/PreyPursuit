from dataclasses import dataclass

@dataclass
class NeuralConfig:
    shift_type: int = 6
    model: str = 'p'
    rbfs: int = 30
    lambda_reg: float = 3