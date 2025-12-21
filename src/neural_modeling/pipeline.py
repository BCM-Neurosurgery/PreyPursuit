from ..patient_data.session import PatientData
from .config import NeuralConfig
from .NGLM import NGLM
import pandas as pd
import arviz as az
import dill as pickle

class NeuralPipeline:
    def __init__(self, patient_data: PatientData, config: NeuralConfig):
        self.patient_data = patient_data
        self.config = config
        self.nglm = NGLM(patient_data, config)
        self.nglm.format_data()
    
    def get_all_nglm_res(self):
        psth = self.patient_data.workspace['psth']
        all_res = []
        for idx, _ in enumerate(psth):
            # fit real model and get coefs/idata
            self.nglm.fit(idx)
            posteriors = self.nglm.sample_posteriors(5000)
            _,_,_, ci_lower, ci_upper = self.nglm.summarize_posterior(posteriors, self.config.credible_interval)
            coefs_keep = self.nglm.coeff_relevance(posteriors, ci_lower, ci_upper)
            idata = self.nglm.compute_idata(posteriors)

            self.nglm.fit_baseline()
            posteriors_base = self.nglm.sample_posteriors(5000, baseline=True)
            idata_base = self.nglm.compute_baseidate(posteriors_base)
            
            # now compare idatas
            comparison = az.compare({'model': idata, 'baseline': idata_base}, ic='waic')
            
            # now append results
            row_info = {
                'comparison': comparison,
                'coefs': coefs_keep,
                'posteriors': posteriors,
                'neuron': idx
            }
            all_res.append(row_info)

        # compile results
        res_df = pd.DataFrame(all_res)
        self.res_df = res_df
    
    def save_results(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.res_df, f)