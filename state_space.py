import pymc as pm
import numpy as np
import aesara.tensor as at
import arviz as az


class DynamicTeamModel:
    """
    Bayesian state-space model for dynamic player-to-team strength mapping.
    Supports AR(1) latent player strengths, optional covariates, and role weights.
    """

    def __init__(self, y, L, t_idx, X=None, role_of_player=None, coords=None):
        """
        Parameters
        ----------
        y : array (M,)
            Observed binary match outcomes (1 = team_i win)
        L : array (M, P)
            Lineup-difference matrix: +w for team_i's players, -w for team_j's players
        t_idx : array (M,)
            Integer index of which time step each match belongs to
        X : array (M, K), optional
            Match covariates (maps, sides, etc.). Default = None.
        role_of_player : array (P,), optional
            Role index per player (if modeling per-role weights). Default = None.
        coords : dict, optional
            Named coordinate structure for nicer ArviZ output
        """
        self.y = np.asarray(y)
        self.L = np.asarray(L)
        self.t_idx = np.asarray(t_idx)
        self.X = np.asarray(X) if X is not None else np.zeros((len(y), 0))
        self.role_of_player = role_of_player
        self.coords = coords or {}
        self.model = None
        self.idata = None

    def build_model(self):
        """Define the PyMC model."""
        y, L, t_idx, X, role_of_player = self.y, self.L, self.t_idx, self.X, self.role_of_player
        P = L.shape[1]
        T = int(t_idx.max()) + 1
        K = X.shape[1]

        coords = self.coords.copy()
        coords.update({
            "match": np.arange(len(y)),
            "player": np.arange(P),
            "time": np.arange(T),
        })
        if K > 0:
            coords["cov"] = np.arange(K)
        if role_of_player is not None:
            coords["role"] = np.unique(role_of_player)

        with pm.Model(coords=coords) as model:
            # ----- Priors -----
            alpha = pm.Normal("alpha", 0.0, 2.0)
            if K > 0:
                beta = pm.Normal("beta", 0.0, 1.0, dims="cov")
            else:
                beta = pm.Deterministic("beta", at.zeros((0,)))

            # AR(1) player dynamics
            sigma0 = pm.HalfNormal("sigma0", 1.0)
            sigma_b = pm.HalfNormal("sigma_b", 0.3)
            phi_raw = pm.Normal("phi_raw", 0.0, 0.5)
            phi = pm.Deterministic("phi", pm.math.tanh(phi_raw))

            b0 = pm.Normal("b_1", 0.0, sigma0, shape=(1, P), dims=("time", "player"))
            b_list = [b0]
            for t in range(1, T):
                bt = pm.Normal(f"b_{t+1}", mu=phi * b_list[-1],
                               sigma=sigma_b, shape=(1, P), dims=("time", "player"))
                b_list.append(bt)
            B = at.concatenate(b_list, axis=0)  # (T, P)

            # Role weighting (optional)
            if role_of_player is not None:
                R = int(np.max(role_of_player)) + 1
                w_role = pm.Normal("w_role", 1.0, 0.3, dims="role")
                W_per_player = w_role[role_of_player]
            else:
                W_per_player = at.ones((P,))

            # Weighted lineup differences
            L_weighted = L * W_per_player
            B_for_matches = B[t_idx, :]  # select correct time for each match
            s_diff = at.sum(L_weighted * B_for_matches, axis=1)

            xb = alpha + (at.dot(X, beta) if K > 0 else 0)
            eta = xb + s_diff

            # ----- Likelihood -----
            pm.Bernoulli("y", logit_p=eta, observed=y, dims="match")

            # ----- Deterministics -----
            pm.Deterministic("p_win", pm.math.sigmoid(eta), dims="match")

        self.model = model
        return model

    def fit(self, draws=1000, tune=1000, target_accept=0.9, random_seed=42, **kwargs):
        """Fit the model via MCMC."""
        if self.model is None:
            self.build_model()
        with self.model:
            self.idata = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=target_accept,
                random_seed=random_seed,
                **kwargs
            )
        return self.idata

    def posterior_summary(self, var_names=None):
        """Return ArviZ summary of posterior samples."""
        if self.idata is None:
            raise RuntimeError("Model not yet fitted. Call .fit() first.")
        return az.summary(self.idata, var_names=var_names)

    def predict(self, L_new, X_new=None, t_idx_new=None):
        """Predict new match outcomes given new design matrices."""
        if self.idata is None:
            raise RuntimeError("Fit model first using .fit().")

        X_new = np.asarray(X_new) if X_new is not None else np.zeros((len(L_new), 0))
        L_new = np.asarray(L_new)
        t_idx_new = np.asarray(t_idx_new) if t_idx_new is not None else np.zeros(len(L_new), int)

        with self.model:
            pm.set_data({"L": L_new, "X": X_new, "t_idx": t_idx_new})
            post_pred = pm.sample_posterior_predictive(self.idata, extend_inferencedata=False)
        return post_pred["y"]