from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import dill as pickle

from .config import Config
from .io import load_behav, load_neural
from .trials import select_trials, subset_trials
from .kinematics import build_kinematics, add_kinematic_features
from .reaction_time import compute_reaction_times
from .trials import cut_to_reaction_time, remove_trials
from .design_matrix import set_larger_prey_first, add_relative_reward


@dataclass
class PatientData:
    patient_id: str
    path: str
    cfg: Config = field(default_factory=Config)

    # state set by .load()
    behav_df: Optional[pd.DataFrame] = None
    trial_df: Optional[pd.DataFrame] = None
    psth: Optional[List] = None
    neuron_info_df: Optional[pd.DataFrame] = None

    # outputs set by .compute_Design_matrix
    kinematics_df: Optional[pd.DataFrame] = None
    design_matrix: Optional[pd.DataFrame] = None
    workspace: Optional[Dict[str, Any]] = None

    def load(self) -> None:
        self.behav_df = load_behav(self.path)
        self.trial_df, self.psth, self.neuron_info_df = load_neural(self.path)

    def compute_design_matrix(self, n_prey: int = 2) -> Dict[str, Any]:
        assert (
            self.behav_df is not None
            and self.trial_df is not None
            and self.psth is not None
        )

        trial_ids = select_trials(self.behav_df, n_prey=n_prey)
        trial_df_sub, psth_sub = subset_trials(self.trial_df, self.psth, trial_ids)

        kin = build_kinematics(
            trial_df_sub,
            rescale=self.cfg.rescale,
            dt=self.cfg.dt,
            smooth=self.cfg.smooth,
        )
        behav_df = self.behav_df
        if n_prey == 2:
            kin, behav_df = set_larger_prey_first(kin, behav_df)

        rts = compute_reaction_times(kin, cfg=self.cfg)
        kin_cut, psth_cut = cut_to_reaction_time(kin, psth_sub, rts)

        kin_feats = add_kinematic_features(kin_cut, n_prey=n_prey)

        # remove paused trials and those with fewer than 10 samples
        _, trials_idcs_kept, kin_clean, psth_clean, behav_clean = remove_trials(
            kin_feats, behav_df.loc[trial_ids], psth_cut
        )

        # if n_prey = 2, ensure larger prey is first
        if n_prey == 2:
            kin_clean, behav_clean = set_larger_prey_first(kin_clean, behav_clean)

        design_mat = add_relative_reward(kin_clean, behav_clean)

        # keep artifacts on the class
        self.kinematics_df = kin_clean
        self.design_matrix = design_mat
        self.workspace = {
            "design_matrix": design_mat,
            "psth": psth_clean,
            "session_variables": behav_clean,
            "kinematics": kin_clean,
            "brain_region": self.neuron_info_df.query("trial_id == 0")[
                "brain_region"
            ].to_numpy(),
            "reaction_times": np.array(
                [rt for idx, rt in enumerate(rts) if idx in trials_idcs_kept]
            ),
        }
        return self.workspace

    def save(self, path: str) -> None:
        if self.workspace is None:
            raise ValueError(
                "No workspace data to save. Please run compute_design_matrix() first."
            )
        with open(path, "wb") as f:
            pickle.dump(self.workspace, f)
