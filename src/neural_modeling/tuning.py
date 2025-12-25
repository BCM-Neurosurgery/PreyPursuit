import pandas as pd
import numpy as np
from typing import Dict, Any
from patsy import dmatrix
import re
from .splines import VAR_NAMES


def calc_tuning_df(glm_df: pd.DataFrame) -> pd.DataFrame:
    tuning_dicts = []
    for _, row in glm_df.iterrows():
        tuning_dict = calc_tuning_row(row)
        tuning_dicts.append(tuning_dict)

    tuning_df = pd.DataFrame(tuning_dicts)
    return tuning_df


# TODO: allow for variiable n_bases and n_interaction bases
def calc_tuning_row(
    glm_row: pd.Series,
    n_steps: int = 10,
    n_bases: int = 11,
    n_interaction_bases: int = 11,
) -> Dict[str, Any]:
    assert n_bases == n_interaction_bases, (
        "n_bases and n_interaction_bases must be equal for now"
    )
    # calculate span for each variable
    spans = {}
    for var in VAR_NAMES:
        var_steps = n_steps if var != "relValue" else 2
        spans[var] = np.linspace(
            glm_row[f"{var}_min"], glm_row[f"{var}_max"], var_steps
        )

    # now create value grid for each var (wt includes relVaue interaction)
    grid_dfs = {}
    for var in VAR_NAMES:
        if var == "relValue":
            continue
        elif var == "wt":
            grids = np.meshgrid(spans["wt"], spans["relValue"])
        else:
            grids = np.meshgrid(spans[var])

        grid_df = {}
        grid_df[var] = grids[0].ravel()
        if var == "wt":
            grid_df["relValue"] = grids[1].ravel()
        grid_df = pd.DataFrame(grid_df)
        grid_dfs[var] = grid_df

    # create basis grids
    basis_grids = {}
    for var in VAR_NAMES:
        if var == "relValue":
            continue
        elif var == "wt":
            formula = f"relValue*cr(wt,df={n_interaction_bases})"
        else:
            formula = f"cr({var},df={n_bases})"

        basis_grid = dmatrix(formula, data=grid_dfs[var], return_type="dataframe")
        basis_grids[var] = basis_grid

    # calculate prediction for each variable
    pred_dict = {}
    for idx, var in enumerate(VAR_NAMES):
        if var == "relValue":
            continue

        # get coefficient relevance, bases and beta values
        coef_mask = glm_row.coefs[f"beta_{idx}"]
        # only keep variable bases
        bases = basis_grids[var]
        bases_to_keep = [
            col for col in bases.columns if re.match(rf"^cr\({var}, df=\d+\)", col)
        ]
        bases = bases[bases_to_keep].copy().values
        betas = glm_row.posteriors_mu[f"beta_{idx}"]

        # get additional coefs for wt interaction
        if var == "wt":
            coef_mask_interaction = glm_row.coefs["beta_tensor_0"]
            bases_interaction = basis_grids[var]
            bases_to_keep_interaction = [
                col for col in bases_interaction.columns if ":" in col
            ]
            bases_interaction = (
                bases_interaction[bases_to_keep_interaction].copy().values
            )
            betas_interaction = glm_row.posteriors_mu["beta_tensor_0"]
            pred = (
                bases @ (coef_mask * betas)
                + bases_interaction @ (coef_mask_interaction * betas_interaction)
            ).reshape(-1, 1)
            pred_dict[var] = pred

            # separately preserve intercept and relValue bases for later linear predictions
            intercept_bases = basis_grids[var]["Intercept"].values
            relValue_bases = basis_grids[var]["relValue"].values
        else:
            pred = (bases @ (coef_mask * betas)).reshape(-1, 1)
            pred_dict[var] = pred

    # now calculate linear prediction terms
    intercept_pred = intercept_bases * glm_row.posteriors_mu["intercept"]
    relValue_pred = (
        relValue_bases
        * glm_row.coefs[f"beta_{idx}"]
        * glm_row.posteriors_mu[f"beta_{idx}"]
    )

    # now calculate tuning prediction for each variable
    pred_means = {var: np.mean(arr) for var, arr in pred_dict.items()}
    tuning_dict = {}
    for var in pred_dict.keys():
        # get means for all other variables except current
        other_means = np.sum([mean for i, mean in pred_means.items() if i != var])

        # and get prediction value for current variables
        if var == "wt":
            pred = pred_dict[var]
        else:
            pred = np.tile(pred_dict[var], (spans["relValue"].shape[0], 1))

        # now calculate tunign curve for variable
        log_pred = (
            pred
            + intercept_pred.reshape(-1, 1)
            + relValue_pred.reshape(-1, 1)
            + other_means
        )
        log_pred = log_pred.reshape(grids[0].shape)
        tuning_pred = (np.exp(log_pred) * 60).T
        tuning_dict[var] = tuning_pred

    return tuning_dict
