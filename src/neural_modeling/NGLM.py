from ..patient_data.session import PatientData
from .config import NeuralConfig
from .design import normalize_data, create_design_matrix
from .models import prs_double_penalty
from numpyro.infer  import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from optax import adam, chain, clip, exponential_decay
from jax.random import PRNGKey

class NGLM:
    def __init__(self, patient_data: PatientData, config: NeuralConfig):
        self.patient_data = patient_data
        self.config = config

    def format_data(self):
        trial_df = self.patient_data.workspace['design_mat']
        trial_wts = self.patient_data.workspace['wt']
        trial_df = normalize_data(trial_df, trial_wts)
        design_mat = create_design_matrix(trial_df)
        self.design_mat = design_mat
        return design_mat

    def fit(self, **kwargs):
        fit_type = self.config.fit_type
        n_steps = self.config.n_steps
        optim = self.config.optim
        guide = self.config.guide
        lrate = self.config.lrate

        if fit_type != 'vi':
            raise NotImplementedError('only variational inference fitting is currently supported')
        if guide != 'normal':
            raise NotImplementedError('only normal guide is currently supported')
        if optim != 'scheduled':
            raise NotImplementedError('only Adam scheduled decay optimizer is currently supported')
        
        # define guide
        guide = AutoNormal(prs_double_penalty)

        # Define an exponential decay schedule for the learning rate
        learning_rate_schedule = exponential_decay(
            init_value=1e-3,  # Starting learning rate
            transition_steps=1000,  # Steps after which the rate decays
            decay_rate=0.9,  # Decay factor for the learning rate
            staircase=True  # If True, the decay happens in steps (discrete) rather than continuous
        )
        svi = SVI(self.model, guide, chain(clip(10.0), adam(learning_rate_schedule)), loss=Trace_ELBO())

        # now run svi algo for model fitting per neuron
        fit_res = []
        for fr_mat in self.workspace['psth']:
            
            res = svi.run(PRNGKey(0), n_steps, bases=self.design_mat['bases'],
                                            base_smoothing_matrix=self.design_mat['base_smoothing_matrix'], 
                                            interaction_tensors=self.design_mat['interaction_tensors'],
                                            tensor_smoothing_matrix=self.design_mat['tensor_smoothing_matrix'],
                                            y=fr_mat, **kwargs)
            fit_res.append(res)
        return fit_res

    def sample_posteriors(self, n_samples=4000):
        fit_res = self.fit_res
            
        posterior_samples = self.config.guide.sample_posterior
        pass

    def compute_idata(self):
        pass
        
    def fit_baseline(self):
        pass

    def compute_baseidate(self):
        pass




