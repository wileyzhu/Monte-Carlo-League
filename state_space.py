import pymc as pm
import aesara.tensor as at
import numpy as np
import arviz as az

class DynamicRoleModel:
    """
    Bayesian dynamic state-space model for evolving player abilities and role impacts.
    """

    def __init__(self, b_obs, L_rows, y, X=None, t_of_match=None, role_of_player=None):
        self.b_obs = np.asarray(b_obs)
        self.L_rows = np.asarray(L_rows)
        self.y = np.asarray(y)
        self.X = np.asarray(X) if X is not None else np.zeros((len(y), 0))
        self.t_of_match = np.asarray(t_of_match) if t_of_match is not None else np.zeros(len(y), dtype=int)
        self.role_of_player = np.asarray(role_of_player) if role_of_player is not None else np.zeros(self.b_obs.shape[1], dtype=int)
        self.model = None
        self.idata = None

    def build_model(self):
        T, P = self.b_obs.shape
        M, K = self.X.shape
        R = len(np.unique(self.role_of_player))

        with pm.Model() as model:
            # --- Global priors ---
            alpha = pm.Normal("alpha", 0, 2)
            beta = pm.Normal("beta", 0, 1, shape=K) if K > 0 else at.zeros((0,))

            # --- AR(1) parameters for ability and weight ---
            phi_b = pm.Beta("phi_b", 9, 1)           # persistence ~ 0.9 mean
            phi_w = pm.Beta("phi_w", 9, 1)
            sigma_b = pm.HalfNormal("sigma_b", 0.5)
            sigma_w = pm.HalfNormal("sigma_w", 0.5)
            sigma_obs = pm.HalfNormal("sigma_obs", 0.3)
            # --- Role baseline impacts ---
            r_role = pm.Normal("r_role", 1.0, 0.3, shape=R)

            # --- Latent initial states ---
            b0 = pm.Normal("b0", 0, 1, shape=P)
            w0 = pm.Normal("w0", mu=r_role[self.role_of_player], sigma=0.3, shape=P)

            # --- Latent evolution ---
            b_true = at.zeros((T, P))
            w_true = at.zeros((T, P))
            b_true = b_true.at[0, :].set(b0)
            w_true = w_true.at[0, :].set(w0)

            for t in range(1, T):
                eps_b = pm.Normal(f"eps_b_{t}", 0, sigma_b, shape=P)
                eps_w = pm.Normal(f"eps_w_{t}", 0, sigma_w, shape=P)
                b_true = b_true.at[t, :].set(phi_b * b_true[t-1, :] + eps_b)
                w_true = w_true.at[t, :].set(phi_w * w_true[t-1, :] + eps_w)
            # --- Measurement model linking observed player scores ---
            
            # Create mask for observed (non-NaN) scores
            obs_mask = ~np.isnan(self.b_obs)
            
            # Flatten arrays and apply mask
            b_true_flat = b_true.flatten()
            b_obs_flat = self.b_obs.flatten()
            obs_idx = np.where(obs_mask.flatten())[0]
            
            # Only model observed scores
            pm.Normal(
                "b_obs",
                mu=b_true_flat[obs_idx],
                sigma=sigma_obs,
                observed=b_obs_flat[obs_idx],
            )
            # --- Observation (match outcomes) ---
            B_for_matches = b_true[self.t_of_match, :]
            W_for_matches = w_true[self.t_of_match, :]
            s_diff = at.sum(self.L_rows * W_for_matches * B_for_matches, axis=1)
            eta = alpha + (self.X @ beta if K > 0 else 0) + s_diff

            pm.Bernoulli("y", logit_p=eta, observed=self.y)
            pm.Deterministic("p_win", pm.math.sigmoid(eta))

        self.model = model
        return model

    def fit(self, draws=1000, tune=1000, target_accept=0.9, seed=42):
        if self.model is None:
            self.build_model()
        with self.model:
            self.idata = pm.sample(draws=draws, tune=tune,
                                   target_accept=target_accept,
                                   random_seed=seed)
        return self.idata

    def summary(self, var_names=None):
        if self.idata is None:
            raise ValueError("Run .fit() first.")
        return az.summary(self.idata, var_names=var_names or
                          ["phi_b", "phi_w", "sigma_b", "sigma_w", "alpha"])

    def predict(self):
        if self.idata is None:
            raise ValueError("Run .fit() first.")
        with self.model:
            ppc = pm.sample_posterior_predictive(self.idata, random_seed=42)
        return ppc