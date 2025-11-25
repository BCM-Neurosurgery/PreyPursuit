from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np
import dill as pickle

from .config import Config
from .io import load_behav, load_neural
from .trials import select_trials, subset_trials
from .kinematics import build_kinematics, add_kinematic_features
from .reaction_time import compute_reaction_times
from .trials import cut_to_reaction_time, remove_trials
from .design_matrix import add_relative_reward


@dataclass
class PatientData:
    patient_id: str
    path: str
    cfg: Config = field(default_factory=Config)

    # state set by .load()
    behav_df: Optional[pd.DataFrame] = None
    trial_df: Optional[pd.DataFrame] = None
    psth: Optional[list] = None
    neuron_info_df: Optional[pd.DataFrame] = None

    # outputs set by .compute_Design_matrix
    kinematics_df: Optional[pd.DataFrame] = None
    design_matrix: Optional[pd.DataFrame] = None
    worksapce: Optional[Dict[str, Any]] = None

    def load(self) -> None:
        self.behav_df = load_behav(self.path)
        self.trial_df, self.psth, self.neuron_info_df = load_neural(self.path)

    def compute_design_matrix(self, n_prey: int = 2) -> Dict[str, Any]:
        assert self.behav_df is not None and self.trial_df is not None and self.psth is not None

        trial_ids = select_trials(self.behav_df, n_prey=n_prey)
        trial_df_sub, psth_sub = subset_trials(self.trial_df, self.psth, trial_ids)

        kin = build_kinematics(trial_df_sub, rescale=self.cfg.rescale, dt=self.cfg.dt, smooth=self.cfg.smooth)
        
        rts = compute_reaction_times(kin, cfg=self.cfg)
        kin_cut, psth_cut = cut_to_reaction_time(kin, psth_sub, rts)

        kin_feats = add_kinematic_features(kin_cut, n_prey=n_prey)

        # remove paused trials and those with fewer than 10 samples
        trials_kept, kin_clean, psth_clean, behav_clean = remove_trials(kin_feats, self.behav_df.loc[trial_ids], psth_cut)

        design_mat = add_relative_reward(kin_clean, behav_clean)

        # keep artifacts on the class
        self.kinematics_df = kin_clean
        self.design_matrix = design_mat
        self.workspace = {
            "design_matrix": design_mat,
            "psth": psth_clean,
            "session_variables": behav_clean,
            "kinematics": kin_clean,
            "brain_region": self.neuron_info_df.query("trial_id == 0")["brain_region"].to_numpy(),
            "reaction_times": np.array([rt for idx, rt in rts if idx in trials_kept]),
        }
        return self.workspace
    
    def save(self, path: str) -> None:
        if self.workspace is None:
            raise ValueError("No workspace data to save. Please run compute_design_matrix() first.")
        with open(path, "wb") as f:
            pickle.dump(self.workspace, f)

class PatientData:
    """
    Class handles all patient data needed for modeling and analysis.
    """
    def __init__(self, patient_id: str, path: str):
        self.patient_id = patient_id
        self.path = path

        # ensure path contains required behavioral data, 
        # optionally include neural data if available, have warning if not
        behav_path  = os.path.join(self.path, "events_info.mat")
        if not os.path.exists(behav_path):
            raise FileNotFoundError(f"Behavioral data not found at {behav_path}")
        else:
            self.behav_path = behav_path
        
        # now load neural data
        neural_path = os.path.join(self.path, "neuronData.mat" )
        if not os.path.exists(neural_path):
            raise FileNotFoundError(f"Neural data not found at {neural_path}.")
        else:
            self.neural_path = neural_path

    def load(self):
        behav_df = self._format_behav_data()
        self.behav_df = behav_df
        trial_df, neuron_data, neuron_info_df = self._format_neural_data()
        self.trial_df = trial_df
        self.neuron_data = neuron_data
        self.neuron_info_df = neuron_info_df

    def compute_design_matrix(self):
        """
        Compute design matrix from behavioral and neural data.
        """
        kinematics = self._get_kinematics()
        psth = self.neuron_data

        # get brain region info from first trail
        trial_mask = self.neuron_info_df['trial_id'] == 0
        brain_regions = self.neuron_info_df[trial_mask]['brain_region'].values

        # subselect N prey trials (default 2)
        trial_ids = self._subselect_trials(self.behav_df, n_prey=2)
        psth = [trial_data for idx, trial_data in enumerate(psth) if idx in trial_ids]
        trial_dfs = []
        kinematic_dfs = []
        for trial in trial_ids:
            trial_mask = self.trial_df['trial_id'] == trial
            to_add = self.trial_df[trial_mask]
            trial_dfs.append(to_add)
            to_add = kinematic_dfs[trial_mask]
            kinematic_dfs.append(to_add)
        self.trial_df = pd.concat(trial_dfs, ignore_index=True)
        kinematics = pd.concat(kinematic_dfs, ignore_index=True)

        # calculate reaction time
        reaction_time = self._get_reaction_time()
        # cut data to reaction time
        kinematics, psth = self._cut_to_reaction_time(kinematics, psth, reaction_time)

        # run additional kinematics processing
        kinematics = self._process_kinematics(kinematics)
        # calculate trials to remove
        trials_to_remove = self._filter_trials()
        # remove trials
        kinematics, psth = self._remove_trials(kinematics, psth, trials_to_remove)
        # add relative reward value to design matrix
        design_matrix = self._add_relative_reward(kinematics)

        # set workspace variable
        workspace = {
            "design_matrix": design_matrix,
            "psth": psth,
            "session_variables": self.behav_df,
            "kinematics": kinematics,
            "brain_region": brain_regions
        }
        self.workspace = workspace
        return workspace


    def _format_behav_data(self):
        behav_raw = loadmat(self.behav_path)
        fieldnames = behav_raw["events_info"].dtype.names
        behav_data = {field: None for field in fieldnames}
        for field in fieldnames:
            field_array = behav_raw["events_info"][field].squeeze()
            behav_data[field] = [item[0].item() for item in field_array]
        return pd.DataFrame(behav_data)
    
    def _format_neural_data(self):
        neural_raw = loadmat(self.neural_path)
        fieldnames = neural_raw["neuronData"].dtype.names
        trial_data = {field: [] for field in fieldnames if field not in {"brain_region", "neruons_info", "spikes"}}
        
        for field in trial_data.keys():
            field_array = neural_raw["neuronData"][field].squeeze()
            for idx, trial in enumerate(field_array):
                trial_array = trial.squeeze()
                n_timestamps = trial_array.shape[0]
                trial_data[field].extend(list(trial_array))
        # now add trial information
        trial_data["trial_id"] = []
        field_array = neural_raw["neuronData"][field].squeeze()
        for idx, trial in enumerate(field_array):
            trial_array = trial.squeeze()
            n_timestamps = trial_array.shape[0]
            trial_data["trial_id"].extend([idx for _ in range(n_timestamps)])

        # create trial dataframe
        trial_df = pd.DataFrame(trial_data)

        # now create neuron matrix
        neuron_data = neural_raw["neuronData"]["spikes"].squeeze()
        neuron_data = [trial_series for trial_series in neuron_data]

        # now create neuron info df
        neuron_info_data = {
            "trial_id": [],
            "neuron_id": [],
            "neuron_label": [],
            "brain_region": []
        }
        # first do for brain region
        field = "brain_region"
        field_array = neural_raw["neuronData"][field].squeeze()
        for trial_idx, trial_data in enumerate(field_array):
            trial_array = trial_data.squeeze()
            n_neurons = trial_array.shape[0]
            neuron_info_data["trial_id"].extend([trial_idx for _ in range(n_neurons)])
            neuron_info_data["brain_region"].extend([region.item() for region in trial_array])
            neuron_info_data["neuron_id"].extend([i for i in range(n_neurons)])
        # now do for neuron labels
        field = "neruons_info"
        field_array = neural_raw["neuronData"][field].squeeze()
        for trial_data in field_array:
            trial_array = trial_data.squeeze()
            neuron_info_data["neuron_label"].extend([label.item() for label in trial_array])
        
        neuron_info_df = pd.DataFrame(neuron_info_data)
        return trial_df, neuron_data, neuron_info_df
    
    def _get_kinematics(self, rescale=0.001, dt=1/60, smooth=False):
        # extract kinematics from trial data
        trial_df = self.trial_df

        df_rows = []
        
        # now iterate through each timestamp and format row correctly for poistion df
        for idx, row in trial_df.iterrows():
            xself = row.x * rescale
            yself = row.y * rescale
            n_prey = 2 if isinstance(row.x_prey, np.ndarray) and row.x_prey.shape[0] == 2 else 1
            if n_prey == 1:
                xprey1 = row.x_prey * rescale
                yprey1 = row.y_prey * rescale
                xprey2 = np.nan
                yprey2 = np.nan
            elif n_prey == 2:
                xprey1 = row.x_prey[0] * rescale
                yprey1 = row.y_prey[0] * rescale
                xprey2 = row.x_prey[1] * rescale
                yprey2 = row.y_prey[1] * rescale
            df_row = {
                'selfXpos': xself,
                'selfYpos': yself,
                'prey1Xpos': xprey1,
                'prey1Ypos': yprey1,
                'prey2Xpos': xprey2,
                'prey2Ypos': yprey2,
            }
            df_rows.append(df_row)

        kinematics_df = pd.DataFrame(df_rows)
        
        # now we need to compute derivatives from the positions!
        for pos_vec in kinematics_df.columns:
            # get vel and acc vector names
            vel_vec = pos_vec.replace('pos', 'vel')
            acc_vec = pos_vec.replace('pos', 'accel')

            # skip empty columns after nanfil
            if np.all(np.isnan(kinematics_df[pos_vec])):
                kinematics_df[vel_vec] = np.full_like(kinematics_df[pos_vec], np.nan)
                kinematics_df[acc_vec] = np.full_like(kinematics_df[pos_vec], np.nan)
                continue
            
            # calculate derivatives and add to kinematics df
            kinematics_df[vel_vec] = np.full_like(kinematics_df[pos_vec], np.nan)
            kinematics_df[acc_vec] = np.full_like(kinematics_df[pos_vec], np.nan)
            for trial_id in trial_df['trial_id'].unique():
                trial_mask = trial_df['trial_id'] == trial_id
                pos_data = kinematics_df.loc[trial_mask, pos_vec].values
                vel_data = np.gradient(pos_data) * 1/dt
                acc_data = np.gradient(vel_data) * 1/dt
                if smooth:
                    vel_data = savgol_filter(vel_data, window_length=11, polyorder=1)
                    acc_data = savgol_filter(acc_data, window_length=11, polyorder=1)
                kinematics_df.loc[trial_mask, vel_vec] = vel_data
                kinematics_df.loc[trial_mask, acc_vec] = acc_data

        kinematics_df['trial_id'] = trial_df['trial_id']
        return kinematics_df
    
    def _subselect_trials(self, n_prey=2):
        trial_df = self.behav_df
        if n_prey not in {1, 2}:
            raise ValueError("n_prey must be 1 or 2.")
        if n_prey == 2:
            trial_ids = trial_df[~np.isnan(trial_df['prey2_val'])].index.values
        else:
            trial_ids = trial_df[np.isnan(trial_df['prey2_val'])].index.values
        return trial_ids
    
    def _get_reaction_time(self, kinematics_df, startidx=0, stopidx=150):
        trial_rts = []
        # get kinematics for each trial
        for trial_id in kinematics_df['trial_id'].unique():
            try:
                trial_mask = kinematics_df['trial_id'] == trial_id
                trial_kinematics = kinematics_df[trial_mask]
                # calculate instantaneous speed (abs difference from initial speed)
                speed = np.sqrt((trial_kinematics['selfXvel'] ** 2) + (trial_kinematics['selfYvel'] ** 2))
                speed = np.abs(speed - speed[0])

                # smooth speed vector
                speed = savgol_filter(speed, window_length=9, polyorder=1)
                # now use ruptures to find change points, only try inspecting start to stop before whole array
                try:
                    algo1 = rpt.Pelt(model="l2").fit(speed[startidx:stopidx])
                    algo2 = rpt.Pelt(model="l1").fit(speed[startidx:stopidx])
                except Exception as e:
                    warnings.warn(f"Ruptures failed to fit speed data for reaction time calculation, using whole time series: {e}")
                    algo1 = rpt.Pelt(model="l2").fit(speed)
                    algo2 = rpt.Pelt(model="l1").fit(speed)

                # retrieve change points w/ .005 penalty
                change_points1 = algo1.predict(pen=0.005)
                change_points2 = algo2.predict(pen=0.005)

                # get first change point as reaction time
                reaction_time = int((change_points1[0] + change_points2[0]) / 2)
                trial_rts.append(reaction_time)
            except Exception as e:
                warnings.warn(f"Failed to compute reaction time for trial {trial_id}: {e}")
                trial_rts.append(np.nan)
        # add as column to behav data
        self.behav_df['reaction_time'] = trial_rts
        return trial_rts
    
    def _cut_to_reaction_time(self, kinematics_df, psth, reaction_time):
        # cut psth first
        psth_cut = []
        for idx, rt in enumerate(reaction_time):
            psth_cut.append(psth[idx][rt-1:])
        # now cut kinematics
        kinematics_cut = []
        for trial_id in kinematics_df['trial_id'].unique():
            trial_mask = kinematics_df['trial_id'] == trial_id
            rt = reaction_time[trial_id]
            trial_kinematics = kinematics_df[trial_mask]
            trial_kinematics = trial_kinematics.iloc[rt-1:]
            kinematics_cut.append(trial_kinematics)
        kinematics_cut_df = pd.concat(kinematics_cut, ignore_index=True)
        return kinematics_cut_df, psth_cut

    def _process_kinematics(self, kinematics_df, n_prey=2):
        # first, add time column to df (and absolute time in ms)
        for trial_id in kinematics_df['trial_id'].unique():
            trial_mask = kinematics_df['trial_id'] == trial_id
            n_timestamps = trial_mask.sum()
            kinematics_df.loc[trial_mask, 'time_samples'] = np.arange(n_timestamps)
            kinematics_df.loc[trial_mask, 'time_ms'] = np.arange(n_timestamps) * (1000/60)  # assuming 60 Hz sampling rate

        # now compute distance from prey
        dx_prey1 = kinematics_df['selfXpos'] - kinematics_df['prey1Xpos']
        dy_prey1 = kinematics_df['selfYpos'] - kinematics_df['prey1Ypos']
        dist_prey1 = np.hypot(dx_prey1, dy_prey1)
        kinematics_df['distPrey1'] = dist_prey1

        if n_prey == 2:
            dx_prey2 = kinematics_df['selfXpos'] - kinematics_df['prey2Xpos']
            dy_prey2 = kinematics_df['selfYpos'] - kinematics_df['prey2Ypos']
            dist_prey2 = np.hypot(dx_prey2, dy_prey2)
            kinematics_df['distPrey2'] = dist_prey2

            # calculate relative distance (closer to prey 1 or 2)
            denom = dist_prey1 + dist_prey2
            # avoid divide by zero
            safe = denom != 0
            reldist = np.empty_like(denom, dtype=float)
            reldist[safe] = (dist_prey1[safe] - dist_prey2[safe]) / denom[safe]
            reldist[~safe] = 0.0
            
            # now save
            kinematics_df['reldistPrey'] = reldist
                                    

        # now compute relative speed
        dx_prey1 = kinematics_df['selfXvel'] - kinematics_df['prey1Xvel']
        dy_prey1 = kinematics_df['selfYvel'] - kinematics_df['prey1Yvel']
        relspeed_prey1 = np.hypot(dx_prey1, dy_prey1)
        kinematics_df['deltaspeedPrey1'] = relspeed_prey1

        if n_prey == 2:
            dx_prey2 = kinematics_df['selfXvel'] - kinematics_df['prey2Xvel']
            dy_prey2 = kinematics_df['selfYvel'] - kinematics_df['prey2Yvel']
            relspeed_prey2 = np.hypot(dx_prey2, dy_prey2)
            kinematics_df['deltaspeedPrey2'] = relspeed_prey2

            # calculate relative speed (closer to prey 1 or 2)
            denom = relspeed_prey1 + relspeed_prey2
            # avoid divide by zero
            safe = denom != 0
            relspeed = np.empty_like(denom, dtype=float)
            relspeed[safe] = (relspeed_prey1[safe] - relspeed_prey2[safe]) / denom[safe]
            relspeed[~safe] = 0.0
            
            # now save
            kinematics_df['relspeed'] = relspeed

        # now compute self speed and angle
        self_speed = np.sqrt(kinematics_df['selfXvel']**2 + kinematics_df['selfYvel']**2)
        self_angle = np.arctan2(kinematics_df['selfYvel'], kinematics_df['selfXvel'])
        kinematics_df['selfSpeed'] = self_speed
        kinematics_df['selfAngle'] = self_angle
        
        return kinematics_df



    
        

                                      



        

    def save(self, path: str):
        if not self.workspace:
            raise ValueError("No workspace data to save. Please run compute_design_matrix() first.")
        with open(path, "wb") as f:
            pickle.dump(self.workspace, f)