# src/patient_data/kinematics.py
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def build_kinematics(trial_df: pd.DataFrame, rescale=1e-3, dt=1/60, smooth=False) -> pd.DataFrame:
    # positions (handles 1 or 2 prey; assumes row.x_prey is scalar or 1D/2D array)
    out = pd.DataFrame({
        "selfXpos": trial_df["x"] * rescale,
        "selfYpos": trial_df["y"] * rescale,
        "trial_id": trial_df["trial_id"].to_numpy(),
    })
    # prey1/prey2
    def _get_col(col, idx=None):
        v = trial_df[col]
        if v.dtype == object:
            if idx is None:
                return v.apply(lambda a: np.nan if not hasattr(a, "shape") else (a * rescale))
            return v.apply(lambda a: np.nan if not hasattr(a, "shape") or a.shape[0] <= idx else a[idx] * rescale)
        return v * rescale
    
    out["prey1Xpos"] = _get_col("x_prey", 0)
    out["prey1Ypos"] = _get_col("y_prey", 0)
    out["prey2Xpos"] = _get_col("x_prey", 1)
    out["prey2Ypos"] = _get_col("y_prey", 1)

    # per-trial derivatives
    for pos_col in ["selfXpos","selfYpos","prey1Xpos","prey1Ypos","prey2Xpos","prey2Ypos"]:
        vel_col = pos_col.replace("pos", "vel")
        acc_col = pos_col.replace("pos", "accel")
        out[[vel_col, acc_col]] = np.nan

        for tid in out["trial_id"].unique():
            m = out["trial_id"] == tid
            pos = out.loc[m, pos_col].to_numpy(dtype=float)
            if np.all(np.isnan(pos)):
                continue
            vel = np.gradient(pos, edge_order=1) / dt
            acc = np.gradient(vel, edge_order=1) / dt
            if smooth:
                vel = savgol_filter(vel, window_length=11, polyorder=1)
                acc = savgol_filter(acc, window_length=11, polyorder=1)
            out.loc[m, vel_col] = vel
            out.loc[m, acc_col] = acc
        
    # time columns per trial
    for tid in out["trial_id"].unique():
        m = out["trial_id"] == tid
        n = int(m.sum())
        idx = np.arange(n)
        out.loc[m, "time_samples"] = idx
        out.loc[m, "time_ms"] = idx * (1000/60)
    
    return out.reset_index(drop=True)

def add_kinematic_features(df: pd.DataFrame, n_prey: int = 2):
    # distances
    dx1 = df["selfXpos"] - df["prey1Xpos"]
    dy1 = df["selfYpos"] - df["prey1Ypos"]
    d1 = np.hypot(dx1, dy1)
    df["distPrey1"] = d1
    if n_prey == 2:
        dx2 = df["selfXpos"] - df["prey2Xpos"]
        dy2 = df["selfYpos"] - df["prey2Ypos"]
        d2 = np.hypot(dx2, dy2)
        denom = d1 + d2
        reldist = np.divide(d1 - d2, denom, out=np.zeros_like(denom, dtype=float), where=(denom != 0))
        df["distPrey2"] = d2
        df["reldistPrey"] = reldist
    
    # rel speed
    dvx1 = df["selfXvel"] - df["prey1Xvel"]
    dvy1 = df["selfYvel"] - df["prey1Yvel"]
    s1 = np.hypot(dvx1, dvy1)
    df["deltaspeedPrey1"] = s1
    if n_prey == 2:
        dvx2 = df["selfXvel"] - df["prey2Xvel"]
        dvy2 = df["selfYvel"] - df["prey2Yvel"]
        s2 = np.hypot(dvx2, dvy2)
        denom = s1 + s2
        rs = np.divide(s1 - s2, denom, out=np.zeros_like(denom, dtype=float), where=(denom != 0))
        df["deltaspeedPrey2"] = s2
        df["relspeed"] = rs

    # self speed
    self_speed = np.hypot(df["selfXvel"], df["selfYvel"])
    self_angle = np.arctan2(df["selfYvel"], df["selfXvel"])
    df["selfSpeed"] = self_speed
    df["selfAngle"] = self_angle
    return df


                                  
