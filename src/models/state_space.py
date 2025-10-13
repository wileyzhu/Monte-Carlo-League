import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
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
    def fit(self):
        """
        Fit Bayesian AR(1) models for each team.
        """
        if self.team_data is None:
            raise ValueError("No team data available. Call prepare_team_data() first.")
            
        print("Fitting Bayesian AR(1) models for each team...")
        
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
                    
                    
                    
                    # Fit Bayesian AR(1) model
                    model, idata = self.create_autoregressive_features(performance_history, p=3)
                    
                    # Store model and inference data
                    self.team_models[team] = {'model': model, 'idata': idata}
                    self.team_histories[team] = performance_history
                    
                    print(f"  Fitted Bayesian AR(1) for {team}: {len(performance_history)} games")
                    
                except Exception as e:
                    print(f"  Failed to fit model for {team}: {str(e)}")
            else:
                print(f"  Insufficient games for {team}: {len(performance_history)} games")
        
        self.fitted = True
        print(f"Fitted Bayesian models for {len(self.team_models)} teams")

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
            # Get model and inference data
            model_data = self.team_models[team]
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
        Get summary of fitted Bayesian models.
        
        Returns:
            DataFrame with team model summaries
        """
        if not self.fitted:
            return pd.DataFrame()
        
        summaries = []
        for team in self.team_models.keys():
            model_data = self.team_models[team]
            idata = model_data['idata']
            history = self.team_histories[team]
            
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
                
                summaries.append({
                    'Team': team,
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