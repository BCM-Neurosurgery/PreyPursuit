from patsy import dmatrix
import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import List


def generate_bases(
    trial_df: pd.DataFrame, n_bases: int = 6, n_interaction_bases: int = 4
) -> tuple[List[pd.DataFrame]]:
    vars = ["speed", "reldistPrey", "relspeed", "rel_samples", "relValue", "wt"]
    bases = []
    for idx, var in enumerate(vars):
        basis = dmatrix(
            f"cr(x{idx + 1}, df=nbases) - 1",
            {f"x{idx + 1}": trial_df[var], "nbases": n_bases},
            return_type="dataframe",
        )
        bases.append(jnp.array(basis.values))

    # get tensor for relative valuet, and multiply with wt to get interaction tensor
    tensor_relval = trial_df["relValue"].to_numpy()[:, np.newaxis]

    basis_interaction = dmatrix(
        "cr(x5x6, df=nbases) - 1",
        {"x5x6": trial_df["wt"], "nbases": n_interaction_bases},
        return_type="dataframe",
    )
    tensor_interaction = pd.DataFrame(basis_interaction * tensor_relval)

    return bases, [jnp.array(tensor_interaction.values)]
