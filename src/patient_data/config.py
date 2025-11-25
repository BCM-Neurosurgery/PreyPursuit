# src/patient_data/config.py
from dataclasses import dataclass

@dataclass
class Config:
    dt: float = 1/60
    rescale: float = 1e-3
    smooth: bool = False
    rt_penalty: float = 0.005
    rt_window: tuple[int, int] = (0, 15)


