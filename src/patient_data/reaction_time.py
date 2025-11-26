# src/patient_data/reaction_time.py
import numpy as np
from scipy.signal import savgol_filter
import ruptures as rpt
import warnings
import pandas as pd
from .config import Config

def compute_reaction_times(kin: pd.DataFrame, cfg: Config) -> list[int | float]:
    rts = []
    for tid in kin["trial_id"].unique():
        try:
            m = kin["trial_id"] == tid
            vx = kin.loc[m, "selfXvel"].to_numpy()
            vy = kin.loc[m, "selfYvel"].to_numpy()
            speed = np.sqrt(vx*vx + vy*vy)
            speed = np.abs(speed - speed[0])
            speed = savgol_filter(speed, cfg.savgol_window, cfg.savgol_poly)

            s, e = cfg.rt_window
            try:
                cp1 = rpt.Pelt(model="l2").fit(speed[s:e]).predict(pen=cfg.rt_penalty)
                cp2 = rpt.Pelt(model="l1").fit(speed[s:e]).predict(pen=cfg.rt_penalty)
            except Exception:
                warnings.warn("PELT on window failed; using full series")
                cp1 = rpt.Pelt(model="l2").fit(speed).predict(pen=cfg.rt_penalty)
                cp2 = rpt.Pelt(model="l1").fit(speed).predict(pen=cfg.rt_penalty)
            rt = int((cp1[0] + cp2[0]) / 2)
            rts.append(rt)
        except Exception as e:
            warnings.warn(f"RT failed trial {tid}: {e}")
            rts.append(np.nan)
    return rts