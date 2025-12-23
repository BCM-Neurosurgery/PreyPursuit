from ..patient_data.session import PatientData
from .config import NeuralConfig
from .NGLM import NGLM
import pandas as pd
import numpy as np
import arviz as az
import dill as pickle
from typing import List, Dict, Any


class NeuralPipeline:
    def __init__(self, patient_data: PatientData, config: NeuralConfig, wt_path: str):
        self.patient_data = patient_data
        self.config = config
        self.nglm = NGLM(patient_data, config, wt_path)
        self.nglm.format_data()

    # run model fitting and stats for each neuron
    def get_all_nglm_res(self) -> pd.DataFrame:
        psth = self.patient_data.workspace["psth"]
        psth = np.vstack(psth)

        # get res for all neurons
        all_res = []
        # results = Parallel(n_jobs=self.config.n_jobs, verbose=100, return_as="generator")(
        #     (delayed(self.get_nglm_res_row)(psth, idx) for idx in range(psth.shape[1]))
        # )
        # for res in results:
        #     all_res.append(res)

        # iterative result for debugging
        for idx in range(psth.shape[1]):
            row_info = self.get_nglm_res_row(psth, idx)
            all_res.append(row_info)

        # compile results
        res_df = pd.DataFrame(all_res)
        self.res_df = res_df

    # get model fit results and stats for individual neuron
    def get_nglm_res_row(self, psth: List[np.ndarray], idx: int) -> Dict[str, Any]:
        try:
            # fit real model and get coefs/idata
            self.nglm.fit(psth, idx)
            posteriors = self.nglm.sample_posteriors(5000)
            _, _, _, ci_lower, ci_upper = self.nglm.summarize_posterior(
                posteriors, self.config.credible_interval
            )
            coefs_keep = self.nglm.coeff_relevance(posteriors, ci_lower, ci_upper)
            idata = self.nglm.compute_idata(posteriors)

            # fit baseline model
            self.nglm.fit_baseline()
            posteriors_base = self.nglm.sample_posteriors(5000, baseline=True)
            idata_base = self.nglm.compute_baseidate(posteriors_base)

            # now compare idatas
            comparison = az.compare({"model": idata, "baseline": idata_base}, ic="waic")

            # now append results
            row_info = {
                "comparison": comparison,
                "coefs": coefs_keep,
                "posteriors": posteriors,
                "neuron": idx,
            }
            return row_info
        except Exception as e:
            print(e)
            row_info = {
                "comparison": None,
                "coefs": None,
                "posteriors": None,
                "neuron": idx,
            }
            return row_info

    # save nueral results
    def save_results(self, output_path):
        with open(output_path, "wb") as f:
            pickle.dump(self.res_df, f)
