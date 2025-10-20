import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pykalman import KalmanFilter
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

def logit(p):
    return np.log(p) - np.log(1 - p)

class TeamAutoregressiveModel:
    """
    Autoregressive model for team performance using average of 5 player scores.
    Tracks team performance over time and predicts future performance.
    """

    def __init__(self, worlds_teams, team_data=None):
        """
        Initialize the autoregressive model.
        
        Args:
            team_data: DataFrame with columns ['Game_ID', 'Team', 'Avg_Performance', 'Win']
            lookback_window: Number of previous games to use for prediction
        """
        self.team_data = team_data.copy() if team_data is not None else None
        self.worlds_teams = worlds_teams
        self.team_models = {}
        self.team_scalers = {}
        self.team_histories = {}
        self.fitted = False

    def prepare_team_data(self, df_with_predictions):
        """
        Prepare team-level data from individual player predictions.
        
        Args:
            df_with_predictions: DataFrame with individual player predictions
            
        Returns:
            DataFrame with team-level performance data
        """
        team_data = []
        
        for game_id in df_with_predictions['Game_ID'].unique():
            game_data = df_with_predictions[df_with_predictions['Game_ID'] == game_id]
            
            if len(game_data) == 10:  # Standard 5v5 match
                # Get team names
                blue_team = game_data['Blue_Team'].iloc[0]
                red_team = game_data['Red_Team'].iloc[0]
                winning_team = game_data['Winning_Team'].iloc[0]
                
                # Split players by team
                winning_players = game_data[game_data['Win'] == 1]
                losing_players = game_data[game_data['Win'] == 0]
                
                if winning_team == blue_team:
                    blue_players = winning_players
                    red_players = losing_players
                else:
                    blue_players = losing_players
                    red_players = winning_players
                
                # Calculate average team performance (mean of 5 player predictions)
                blue_avg = blue_players['prediction'].mean()
                red_avg = red_players['prediction'].mean()
                
                # Add team data
                team_data.append({
                    'Game_ID': game_id,
                    'Team': blue_team,
                    'Avg_Performance': blue_avg,
                    'Win': 1 if winning_team == blue_team else 0
                })
                
                team_data.append({
                    'Game_ID': game_id,
                    'Team': red_team,
                    'Avg_Performance': red_avg,
                    'Win': 1 if winning_team == red_team else 0
                })
        
        self.team_data = pd.DataFrame(team_data)
        return self.team_data

    def create_autoregressive_features(self, data, p: int = 1, **priors):
        """
        Create a PyMC Bayesian AR(p) model (default AR(1)) from team performance history.
        
        Args:
            data: List or array of performance scores in chronological order
            p: Number of autoregressive lags (e.g., 1 for AR(1), 3 for AR(3))
            **priors: Dictionary containing prior specifications
            
        Returns:
            (model, inference_data): The PyMC model and its inference results
        """
        # --- Default priors with decay structure ---
        decay = 0.6  # how fast the mean shrinks with lag order
        means = [0.0] + [0.5 * (decay ** (i-1)) for i in range(1, p + 1)]
        
        # Exponential decay in standard deviations for AR coefficients
        sigmas = 0.5 * np.exp(-np.linspace(0, 1, p))  # exponential decay in sd
        coef_sigmas = [1.0] + sigmas.tolist()  # constant term gets higher variance
        
        default_priors = {
            # [constant term, φ₁...φ_p] → total size = p + 1
            'coefs': {'mu': means, 'sigma': coef_sigmas, 'size': p + 1},
            'sigma': 0.5,
            # initial state: first p values to start recursion
            'init': {'mu': 0.5, 'sigma': 0.3, 'size': p}
        }

        # Merge user priors with defaults
        for key, val in default_priors.items():
            priors.setdefault(key, val)

        # Ensure data is array
        data = np.asarray(data, dtype=float)
        n_obs = len(data)
        if n_obs <= p:
            raise ValueError(f"Need more than {p} observations for AR({p}) model.")

        # Build PyMC model
        with pm.Model() as AR:
            # Coefficients: α (constant) and φ₁..φ_p
            coefs = pm.Normal(
                "coefs",
                mu=priors['coefs']['mu'],
                sigma=priors['coefs']['sigma'],
                shape=priors['coefs']['size']
            )

            # Noise
            sigma = pm.HalfNormal("sigma", sigma=priors['sigma'])

            # Initial state for first p lags
            init = pm.Normal.dist(
                priors['init']['mu'],
                priors['init']['sigma'],
                shape=priors['init']['size']
            )

            # AR(p) process
            ar_p = pm.AR(
                "ar",
                coefs,
                sigma=sigma,
                init_dist=init,
                constant=True,
                steps=n_obs - p,   # number of steps to generate after init
            )

            # Likelihood - ensure shapes match
            observed_data = data[p:]
            pm.Normal("likelihood", mu=ar_p[:len(observed_data)], sigma=sigma, observed=observed_data)

            # --- Sampling ---
            idata_ar = pm.sample_prior_predictive(samples=500, random_seed=42)
            idata_ar.extend(pm.sample(draws=1000, tune=1000, target_accept=0.9,
                                    random_seed=42, return_inferencedata=True))
            pm.sample_posterior_predictive(idata_ar, extend_inferencedata=True, random_seed=42)

        return AR, idata_ar
        
    def kalman_filters(self, data, obs_noise_multiplier=2.0, transition_ratio=0.01):
        """
        Apply Kalman Filter to track team performance over time.
        
        Args:
            data: Array of performance observations (0-1 scale)
            obs_noise_multiplier: Multiplier for observation covariance (default 1.5)
                Higher = less reactive to individual games (more stable)
                Lower = more reactive to recent performance (more volatile)
            transition_ratio: Ratio of transition to observation covariance (default 0.02)
                Higher = skill can change faster (more reactive)
                Lower = skill changes slowly (more stable)
            
        Returns:
            tuple: (state_means, state_covariances) - filtered estimates
            
        Parameter Selection Guide:
        - transition_matrices: [1] assumes skill persists (random walk)
        - observation_matrices: [1] assumes we directly observe skill
        - initial_state_mean: Start at data mean (average performance)
        - initial_state_covariance: 0.25 (moderate initial uncertainty)
        - observation_covariance: data_var * obs_noise_multiplier
        - transition_covariance: obs_cov * transition_ratio
        
        Tuning Tips:
        - For MORE stability (less reactive): increase obs_noise_multiplier, decrease transition_ratio
        - For MORE reactivity (track recent changes): decrease obs_noise_multiplier, increase transition_ratio
        - Recommended stable settings: obs_noise_multiplier=1.5-2.0, transition_ratio=0.01-0.03
        - Recommended reactive settings: obs_noise_multiplier=1.0, transition_ratio=0.05-0.10
        """
        # Calculate data statistics for parameter tuning
        data_array = np.array(data)
        data_mean = np.mean(data_array)
        data_var = np.var(data_array)
        
        # Parameter selection based on data characteristics:
        # 1. Observation covariance: Treat games as noisy measurements
        #    HIGHER value = trust individual games less, smoother predictions
        #    We want HIGH observation noise to reduce reactivity to outliers
        obs_cov = max(data_var * obs_noise_multiplier, 0.1)  # Increased minimum to 0.1
        
        # 2. Transition covariance: How much skill can change between games
        #    LOWER value = skill changes very slowly, more stable
        #    We want VERY LOW transition noise for stability
        trans_cov = max(obs_cov * transition_ratio, 0.001)  # Very small, minimum 0.001
        
        # 3. Initial state: Use data mean or 0.5 (neutral performance)
        init_mean = data_mean if not np.isnan(data_mean) else 0.5
        
        # 4. Initial covariance: Moderate uncertainty
        init_cov = 0.1
        
        kf = KalmanFilter(
            transition_matrices=[1],           # State persists (random walk)
            observation_matrices=[1],          # Direct observation of state
            initial_state_mean=init_mean,      # Start at data mean
            initial_state_covariance=init_cov, # Moderate initial uncertainty
            observation_covariance=obs_cov,    # Measurement noise from data
            transition_covariance=trans_cov    # Small skill changes (10% of obs noise)
        )
        
        # Use smoothing instead of filtering for better estimates
        # Smoothing uses all data (past and future) to estimate each state
        # This reduces reactivity to outliers and gives more stable predictions
        state_means_smooth, state_covariances_smooth = kf.smooth(data_array)
        
        return state_means_smooth, state_covariances_smooth

    def fit(self, model_type='bayesian'):
        """
        Fit time series models for each team.
        
        Args:
            model_type: Type of model to fit
                - 'bayesian': Bayesian AR(3) model (default)
                - 'kalman_filter': Kalman Filter for state tracking
        """
        if self.team_data is None:
            raise ValueError("No team data available. Call prepare_team_data() first.")
        
        # Store the model type
        self.model_type = model_type
        
        if model_type == 'bayesian':
            print("Fitting Bayesian AR(3) models for each team...")
        elif model_type == 'kalman_filter':
            print("Fitting Kalman Filter models for each team...")
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'bayesian' or 'kalman_filter'.")
        
        # Group by team and sort by game chronologically
        for team in self.worlds_teams:
            team_games = self.team_data[self.team_data['Team'] == team].copy()
            team_games = team_games.sort_values('Game_ID')
            
            # Get performance history
            performance_history = team_games['Avg_Performance'].tolist()
            
            if len(performance_history) >= 5:  # Minimum data for AR(1)
                try:
                    # Define priors based on data characteristics
                    data_mean = np.mean(performance_history)
                    data_std = np.std(performance_history)
                    
                    
                    if model_type == 'bayesian':
                        # Fit Bayesian AR(3) model
                        model_name, idata = self.create_autoregressive_features(performance_history, p=3)
                        
                        # Store model and inference data
                        self.team_models[team] = {
                            'model_type': 'bayesian',
                            'model': model_name,
                            'idata': idata
                        }
                        self.team_histories[team] = performance_history
                        
                        print(f"  Fitted Bayesian AR(3) for {team}: {len(performance_history)} games")
                    
                    elif model_type == 'kalman_filter':
                        # Fit Kalman Filter
                        state_means, state_covs = self.kalman_filters(performance_history)
                        
                        self.team_models[team] = {
                            'model_type': 'kalman_filter',
                            'state_means': state_means,
                            'state_covariances': state_covs
                        }
                        self.team_histories[team] = performance_history
                        
                        print(f"  Fitted Kalman Filter for {team}: {len(performance_history)} games")
                    
                except Exception as e:
                    print(f"  Failed to fit model for {team}: {str(e)}")
            else:
                print(f"  Insufficient games for {team}: {len(performance_history)} games")
        
        self.fitted = True
        model_name = 'Bayesian AR(3)' if model_type == 'bayesian' else 'Kalman Filter'
        print(f"Fitted {model_name} models for {len(self.team_models)} teams")

    def predict_team_performance(self, team, recent_performances=None, n_samples=100):
        """
        Predict next performance for a team using Bayesian AR(1) model.
        
        Args:
            team: Team name
            recent_performances: List of recent performance scores (optional)
            n_samples: Number of posterior samples for prediction
            
        Returns:
            Dictionary with mean prediction and uncertainty bounds
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if team not in self.team_models:
            # Return average performance if no model available
            if team in self.team_histories:
                avg_perf = np.mean(self.team_histories[team])
                return {
                    'mean': avg_perf,
                    'std': np.std(self.team_histories[team]),
                    'lower_95': avg_perf - 1.96 * np.std(self.team_histories[team]),
                    'upper_95': avg_perf + 1.96 * np.std(self.team_histories[team])
                }
            else:
                return {'mean': 0.5, 'std': 0.2, 'lower_95': 0.1, 'upper_95': 0.9}
        
        # Use provided recent performances or team history
        if recent_performances is not None:
            performance_history = recent_performances
        else:
            performance_history = self.team_histories[team]
        
        if len(performance_history) < 3:
            # Not enough history, return average
            avg_perf = np.mean(performance_history) if performance_history else 0.5
            std_perf = np.std(performance_history) if len(performance_history) > 1 else 0.2
            return {
                'mean': avg_perf,
                'std': std_perf,
                'lower_95': avg_perf - 1.96 * std_perf,
                'upper_95': avg_perf + 1.96 * std_perf
            }
        
        try:
            # Get model data
            model_data = self.team_models[team]
            model_type = model_data.get('model_type', 'bayesian')
            
            if model_type == 'kalman_filter':
                # For Kalman Filter, use the last filtered state as prediction
                state_means = model_data['state_means']
                state_covs = model_data['state_covariances']
                
                # Last state is the best prediction
                mean_pred = state_means[-1][0]
                std_pred = np.sqrt(state_covs[-1][0, 0])
                
                return {
                    'mean': np.clip(mean_pred, 0.0, 1.0),
                    'std': std_pred,
                    'lower_95': np.clip(mean_pred - 1.96 * std_pred, 0.0, 1.0),
                    'upper_95': np.clip(mean_pred + 1.96 * std_pred, 0.0, 1.0)
                }
            
            else:  # Bayesian model
                model = model_data['model']
                idata = model_data['idata']
                
                # Use PyMC's standard posterior predictive sampling
                with model:
                    ppc = pm.sample_posterior_predictive(idata)
                
                # Extract predictions from the likelihood
                predictions = ppc.posterior_predictive['likelihood'].values.flatten()
            
            # Calculate statistics
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            lower_95 = np.percentile(predictions, 2.5)
            upper_95 = np.percentile(predictions, 97.5)
            
            # Clip to reasonable range
            return {
                'mean': np.clip(mean_pred, 0.0, 1.0),
                'std': std_pred,
                'lower_95': np.clip(lower_95, 0.0, 1.0),
                'upper_95': np.clip(upper_95, 0.0, 1.0)
            }
            
        except Exception as e:
            print(f"Prediction failed for {team}: {str(e)}")
            # Fallback to simple average
            avg_perf = np.mean(performance_history)
            std_perf = np.std(performance_history)
            return {
                'mean': avg_perf,
                'std': std_perf,
                'lower_95': avg_perf - 1.96 * std_perf,
                'upper_95': avg_perf + 1.96 * std_perf
            }

    def predict_match_outcome(self, team1, team2, team1_recent=None, team2_recent=None):
        """
        Predict match outcome between two teams using Bayesian predictions.
        
        Args:
            team1: First team name
            team2: Second team name
            team1_recent: Recent performance history for team1 (optional)
            team2_recent: Recent performance history for team2 (optional)
            
        Returns:
            Dictionary with team performance predictions and win probabilities
        """
        team1_pred = self.predict_team_performance(team1, team1_recent)
        team2_pred = self.predict_team_performance(team2, team2_recent)
        
        # Use mean predictions for win probability calculation
        team1_perf = team1_pred['mean']
        team2_perf = team2_pred['mean']
        
        # Calculate win probability using softmax on performance scores
        # Clip to avoid numerical issues
        team1_perf_clipped = np.clip(team1_perf, 0.001, 0.999)
        team2_perf_clipped = np.clip(team2_perf, 0.001, 0.999)
        
        # Convert to logits for softmax calculation
        logit1 = logit(team1_perf_clipped)
        logit2 = logit(team2_perf_clipped)
        
        # Apply softmax to get win probabilities
        max_logit = max(logit1, logit2)  # For numerical stability
        exp1 = np.exp(logit1 - max_logit)
        exp2 = np.exp(logit2 - max_logit)
        
        team1_win_prob = exp1 / (exp1 + exp2)
        
        return {
            'team1_performance': team1_pred,
            'team2_performance': team2_pred,
            'team1_win_probability': team1_win_prob,
            'team2_win_probability': 1 - team1_win_prob,
            'confidence': 1 - min(team1_pred['std'], team2_pred['std'])  # Lower std = higher confidence
        }

    def get_team_summary(self):
        """
        Get summary of fitted time series models.
        
        Returns:
            DataFrame with team model summaries
        """
        if not self.fitted:
            return pd.DataFrame()
        
        summaries = []
        for team in self.team_models.keys():
            model_data = self.team_models[team]
            model_type = model_data.get('model_type', 'bayesian')
            history = self.team_histories[team]
            
            # Get predicted mean for next performance
            try:
                team_pred = self.predict_team_performance(team)
                predicted_mean = team_pred['mean']
                predicted_std = team_pred['std']
                predicted_lower = team_pred['lower_95']
                predicted_upper = team_pred['upper_95']
            except:
                # Fallback if prediction fails
                predicted_mean = np.mean(history)
                predicted_std = np.std(history)
                predicted_lower = predicted_mean - 1.96 * predicted_std
                predicted_upper = predicted_mean + 1.96 * predicted_std
            
            if model_type == 'kalman_filter':
                # For Kalman Filter, extract state statistics
                state_means = model_data['state_means']
                state_covs = model_data['state_covariances']
                
                # Get final state estimate
                final_state_mean = state_means[-1][0]
                final_state_std = np.sqrt(state_covs[-1][0, 0])
                
                summaries.append({
                    'Team': team,
                    'Model_Type': 'Kalman Filter',
                    'Games_Played': len(history),
                    'Avg_Performance': np.mean(history),
                    'Performance_Std': np.std(history),
                    'Final_State_Mean': final_state_mean,
                    'Final_State_Std': final_state_std,
                    'Predicted_Mean': predicted_mean,
                    'Predicted_Std': predicted_std,
                    'Predicted_Lower_95': predicted_lower,
                    'Predicted_Upper_95': predicted_upper
                })
                
            else:  # Bayesian model
                idata = model_data['idata']
                
                # Extract posterior statistics for AR(3) model
                try:
                    coef_summary = az.summary(idata, var_names=['coefs'])
                    sigma_mean = az.summary(idata, var_names=['sigma'])['mean'].iloc[0]
                    
                    # Extract all AR(3) coefficients: [constant, φ₁, φ₂, φ₃]
                    constant_mean = coef_summary['mean'].iloc[0]  # Constant term
                    ar1_coef_mean = coef_summary['mean'].iloc[1]  # AR lag 1 coefficient
                    ar2_coef_mean = coef_summary['mean'].iloc[2]  # AR lag 2 coefficient  
                    ar3_coef_mean = coef_summary['mean'].iloc[3]  # AR lag 3 coefficient
                    
                    constant_std = coef_summary['sd'].iloc[0]
                    ar1_coef_std = coef_summary['sd'].iloc[1]
                    ar2_coef_std = coef_summary['sd'].iloc[2]
                    ar3_coef_std = coef_summary['sd'].iloc[3]
                    
                    summaries.append({
                        'Team': team,
                        'Model_Type': 'Bayesian AR(3)',
                        'Games_Played': len(history),
                        'Avg_Performance': np.mean(history),
                        'Performance_Std': np.std(history),
                        'Constant_Mean': constant_mean,
                        'Constant_Std': constant_std,
                        'AR1_Coef_Mean': ar1_coef_mean,
                        'AR1_Coef_Std': ar1_coef_std,
                        'AR2_Coef_Mean': ar2_coef_mean,
                        'AR2_Coef_Std': ar2_coef_std,
                        'AR3_Coef_Mean': ar3_coef_mean,
                        'AR3_Coef_Std': ar3_coef_std,
                        'Noise_Sigma': sigma_mean,
                        'Predicted_Mean': predicted_mean,
                    'Predicted_Std': predicted_std,
                    'Predicted_Lower_95': predicted_lower,
                        'Predicted_Upper_95': predicted_upper
                    })
                except Exception as e:
                    summaries.append({
                        'Team': team,
                        'Model_Type': 'Bayesian AR(3)',
                        'Games_Played': len(history),
                        'Avg_Performance': np.mean(history),
                        'Performance_Std': np.std(history),
                        'Constant_Mean': np.nan,
                        'Constant_Std': np.nan,
                        'AR1_Coef_Mean': np.nan,
                        'AR1_Coef_Std': np.nan,
                        'AR2_Coef_Mean': np.nan,
                        'AR2_Coef_Std': np.nan,
                        'AR3_Coef_Mean': np.nan,
                        'AR3_Coef_Std': np.nan,
                        'Noise_Sigma': np.nan,
                        'Predicted_Mean': np.mean(history),
                        'Predicted_Std': np.std(history),
                        'Predicted_Lower_95': np.nan,
                        'Predicted_Upper_95': np.nan
                    })
        
        return pd.DataFrame(summaries)