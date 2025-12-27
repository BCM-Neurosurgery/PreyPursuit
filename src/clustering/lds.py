import numpy as np
from ssm import LDS
from ssm.lds import LDS as LDSObj
from ssm.variational import SLDSMeanFieldVariationalPosterior as Q
from typing import List, Tuple, Any


def calc_lds(
    X_train: List[np.ndarray], reldist_inputs: List[np.ndarray]
) -> Tuple[LDSObj, np.ndarray, Q]:
    # create lds object
    lds = LDS(X_train[0].shape[1], 3, M=1, emissions="gaussian")
    # fit to our fr data
    elbos, q = lds.fit(
        X_train,
        reldist_inputs,
        method="bbvi",
        variational_posterior="meanfield",
        num_iters=3000,
    )

    # fit and return
    return lds, elbos, q
