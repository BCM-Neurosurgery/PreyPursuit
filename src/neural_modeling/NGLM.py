from ..patient_data.session import PatientData
from .config import NeuralConfig
from .design import normalize_data, create_design_matrix
from .models import prs_double_penalty, baseline_noise_model
import numpy as np
import arviz as az
import optim
import numpyro.distributions as dist
from numpyro.infer  import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from optax import adam, chain, clip, exponential_decay
from jax.random import PRNGKey
import jax.numpy as jnp

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

    def fit(self, neuron_idx, **kwargs):
        fit_type = self.config.fit_type
        n_steps = self.config.n_steps
        optimizer = self.config.optimizer
        guide = self.config.guide

        if fit_type != 'vi':
            raise NotImplementedError('only variational inference fitting is currently supported')
        if guide != 'normal':
            raise NotImplementedError('only normal guide is currently supported')
        if optimizer != 'scheduled':
            raise NotImplementedError('only Adam scheduled decay optimizer is currently supported')
        
        # define guide
        guide = AutoNormal(prs_double_penalty)
        self.guide = guide

        # Define an exponential decay schedule for the learning rate
        learning_rate_schedule = exponential_decay(
            init_value=1e-3,  # Starting learning rate
            transition_steps=1000,  # Steps after which the rate decays
            decay_rate=0.9,  # Decay factor for the learning rate
            staircase=True  # If True, the decay happens in steps (discrete) rather than continuous
        )
        svi = SVI(prs_double_penalty, guide, chain(clip(10.0), adam(learning_rate_schedule)), loss=Trace_ELBO())

        # now run svi algo for model fitting per neuron
        fr_mat = self.workspace['psth'][neuron_idx]
        self.y = fr_mat
      
        res = svi.run(PRNGKey(0), n_steps, bases=self.design_mat['bases'],
                                        base_smoothing_matrix=self.design_mat['base_smoothing_matrix'], 
                                        interaction_tensors=self.design_mat['interaction_tensors'],
                                        tensor_smoothing_matrix=self.design_mat['tensor_smoothing_matrix'],
                                        y=fr_mat, **kwargs)
        
        self.fit_res = res
        return res

    def sample_posteriors(self, n_samples=4000, baseline=False):
        fit_res = self.fit_res if not baseline else self.noise_res
        guide = self.guide if not baseline else self.noise_guide
            
        posterior_samples = guide.sample_posterior(PRNGKey(1), fit_res.params, sample_shape=(n_samples,))
        posterior_samples = {key: posterior_samples[key] for key in posterior_samples if
                                  key.startswith('beta_') or key.startswith('intercept')}
        self.posterior_samples = posterior_samples
        self.n_post_samples = n_samples
        return posterior_samples
    
    def summarize_posterior(self, posterior_samples, credible_interval=90):
        lower = 0 + int((100-credible_interval)/2)
        upper = 100 - int((100-credible_interval)/2)

        self.posterior_means = {}
        self.posterior_medians = {}
        self.posterior_sd = {}
        self.posterior_ci_lower = {}
        self.posterior_ci_upper = {}

        for keys in posterior_samples.keys():
            self.posterior_means[keys] = jnp.mean(posterior_samples[keys], axis=0)
            self.posterior_medians[keys] = jnp.median(posterior_samples[keys], axis=0)
            self.posterior_sd[keys] = jnp.std(posterior_samples[keys], axis=0)
            self.posterior_ci_lower[keys] = jnp.percentile(posterior_samples[keys], lower, axis=0)
            self.posterior_ci_upper[keys] = jnp.percentile(posterior_samples[keys], upper, axis=0)
        
        return (self.posterior_means, self.posterior_medians,
                self.posterior_sd, self.posterior_ci_lower,
                self.posterior_ci_upper)
    
    def coeff_relevance(self, posterior_samples, ci_upper, ci_lower):
        self.coef_keep = {}

        for keys in posterior_samples.keys():
            self.coef_keep[keys] = np.logical_xor(ci_lower[keys]>0, ci_upper[keys]<0).astype(int)

        return self.coef_keep

    def compute_idata(self, posterior_samples):
        pointwise_log_likelihood = []
        basis_keys = [key for key in posterior_samples if key.startswith('beta_beta_')]
        tensor_keys = [key for key in posterior_samples if key.startswith('beta_tensor_')]
        intercept_keys = [key for key in posterior_samples if key.startswith('intercept')]
        n_samples = posterior_samples[basis_keys[0]].shape[0]

        for i in range(n_samples):

            # add contributions from bases
            for j, key in enumerate(basis_keys):
                if j == 0:
                    linear_pred = jnp.dot(posterior_samples[key][i], self.design_mat['bases'][j].transpose())
                else:
                    linear_pred += jnp.dot(posterior_samples[key][i], self.design_mat['bases'][j].transpose())

            # add contributions from tensors
            for j, key in enumerate(tensor_keys):
                linear_pred += jnp.dot(posterior_samples[key][i], self.design_mat['interaction_tensors'][j].transpose())

            # add contribution from intercept
            for j, key in enumerate(intercept_keys):
                linear_pred += posterior_samples[key][i]

            # Calculate log-likelihood for each data point under a Poisson likelihood
            log_likelihood = dist.Poisson(rate=jnp.exp(linear_pred)).log_prob(self.y)
            pointwise_log_likelihood.append(log_likelihood)

        pointwise_log_likelihood = jnp.array(pointwise_log_likelihood)
        # # Convert pointwise log-likelihood to ArviZ's InferenceData format
        # Since it's vi expand dimension to emulate a chain
        pointwise_log_likelihood = pointwise_log_likelihood[None, :, :]
        idata = az.from_dict(log_likelihood={"log_likelihood": pointwise_log_likelihood})
        return idata
        
    def fit_baseline(self):
        fr_mat = self.y
        self.noise_guide = AutoNormal(baseline_noise_model)
        optimizer = optim.ClippedAdam(step_size = 1e-2)

        svi = SVI(baseline_noise_model, self.noise_guide, optimizer, loss=Trace_ELBO())
        self.noise_res = svi.run(PRNGKey(0), 2000, y=jnp.array(fr_mat))
        return self.noise_res

    def compute_baseidate(self, posterior_samples):
        n_samples = posterior_samples['intercept'].shape[0]
        y_len = self.y.shape[0]

        for i in range(n_samples):
            linear_pred = jnp.exp(jnp.repeat(posterior_samples['intercept'][i], y_len))
            log_likelihood = dist.Poisson(rate=linear_pred).log_prob(self.y)
            pointwise_log_likelihood.append(log_likelihood)

        pointwise_log_likelihood = jnp.array(pointwise_log_likelihood)
        pointwise_log_likelihood = pointwise_log_likelihood[None, :, :]
        idata = az.from_dict(log_likelihood={"log_likelihood": pointwise_log_likelihood})
        return idata





