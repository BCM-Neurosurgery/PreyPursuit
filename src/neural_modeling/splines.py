from patsy import dmatrix
import numpy as np
import pandas as pd

def generate_bases(trial_df, n_bases=6, n_interaction_bases=4):
    vars = ['speed', 'reldist', 'relspeed', 'reltime', 'relValue', 'wt']
    bases = {}
    for idx, var in enumerate(vars):
        basis = dmatrix(f"cr(x{idx + 1}, df=nbases) - 1", {f"x{idx + 1}": trial_df[var], "nbases": n_bases},
                             return_type="dataframe")
        bases[var] = basis
    
    # if generate interactions, get interaction tensors
    bases_interactions = {}
    for idx, var in vars[:-1]:
        basis_interaction = dmatrix(f"cr(x{idx + 1}, df=nbases) - 1", {f"x{idx + 1}": trial_df[var], "nbases": n_interaction_bases},
                            return_type="dataframe")
        bases_interactions[var] = basis_interaction

    # get tensor for wt, and multiply with relative value to get interactin tensor
    tensor_wt = trial_df['wt'].to_numpy()[:, np.newaxis]
    
    basis_interaction = dmatrix(f"cr(x5, df=nbases) - 1", {f"x5": trial_df['relValue'], "nbases": n_interaction_bases},
                            return_type="dataframe")
    tensor_interaction = pd.DataFrame(basis_interaction * tensor_wt)

    return bases, tensor_interaction

        




    

