#!/usr/bin/env python3
"""
Test script for the Bayesian AR(1) model
"""
import numpy as np
import pandas as pd
from state_space import TeamAutoregressiveModel

# Create some sample team data
np.random.seed(42)
sample_data = []

teams = ['Team_A', 'Team_B', 'Team_C']
for game_id in range(1, 21):  # 20 games
    for i, team in enumerate(teams[:2]):  # Only use 2 teams per game
        # Simulate team performance with some autocorrelation
        base_perf = 0.5 + 0.1 * np.sin(game_id * 0.3) + np.random.normal(0, 0.1)
        performance = np.clip(base_perf, 0.1, 0.9)
        
        sample_data.append({
            'Game_ID': game_id,
            'Team': team,
            'Avg_Performance': performance,
            'Win': 1 if i == 0 else 0  # Team_A wins odd games, Team_B wins even
        })

# Create DataFrame
team_data = pd.DataFrame(sample_data)
print("Sample team data:")
print(team_data.head(10))
print(f"\nTotal games: {len(team_data)}")

# Initialize and fit the model
print("\n" + "="*50)
print("Testing Bayesian AR(1) Model")
print("="*50)

model = TeamAutoregressiveModel(team_data=team_data, lookback_window=3)

try:
    # Fit the model
    model.fit()
    
    # Get team summary
    print("\nTeam Model Summary:")
    summary = model.get_team_summary()
    print(summary)
    
    # Test predictions
    print("\n" + "-"*30)
    print("Testing Predictions")
    print("-"*30)
    
    # Predict team performance
    team_a_pred = model.predict_team_performance('Team_A')
    print(f"\nTeam_A next performance prediction:")
    print(f"  Mean: {team_a_pred['mean']:.3f}")
    print(f"  Std: {team_a_pred['std']:.3f}")
    print(f"  95% CI: [{team_a_pred['lower_95']:.3f}, {team_a_pred['upper_95']:.3f}]")
    
    # Predict match outcome
    match_pred = model.predict_match_outcome('Team_A', 'Team_B')
    print(f"\nMatch prediction (Team_A vs Team_B):")
    print(f"  Team_A win probability: {match_pred['team1_win_probability']:.3f}")
    print(f"  Team_B win probability: {match_pred['team2_win_probability']:.3f}")
    print(f"  Confidence: {match_pred['confidence']:.3f}")
    
    print("\n✅ Bayesian AR(1) model test completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error during testing: {str(e)}")
    import traceback
    traceback.print_exc()