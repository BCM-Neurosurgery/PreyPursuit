import numpy as np

REL_VALUES = 4
SAMPLING_RATE = 60

def normalize_data(trial_df, trial_wts):
    # normalize 0-1 for relative reward
    trial_df['relValue'] = trial_df['relValue'] / REL_VALUES

    # subtract average relative distance
    trial_df['reldistPrey'] -= trial_df['reldistPrey'].mean()

    # recalculate relative speed to account for prey values
    dspeed1_value_normalized = trial_df['deltaspeedPrey1'] / trial_df['val1']
    dspeed2_value_normalized = trial_df['deltaspeedPrey2'] / trial_df['val2']
    value_normalized_speed_diff = dspeed1_value_normalized - dspeed2_value_normalized

    # now min-max normalize this series to get new relative speed (and then also subtract mean)
    trial_df['relspeed'] = (value_normalized_speed_diff - value_normalized_speed_diff.min()) / (value_normalized_speed_diff.max() - value_normalized_speed_diff.min())
    trial_df['relspeed'] -= trial_df['relspeed'].mean()

    # now subtract mean timestamp to get relative time
    trial_df['reltime'] -= trial_df['reltime'].mean()

    # subtract mean self speed to normalize
    trial_df['speed'] -= trial_df['selfSpeed'].mean()

    # add wt value to trial df
    trial_df['wt'] = np.vstack(trial_wts) - 0.5
    
    return trial_df

def create_design_matrix(trial_df, n_bases=6, n_interaction_bases=4):
    pass

