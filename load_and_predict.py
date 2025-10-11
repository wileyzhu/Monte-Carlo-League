"""
Model Loading and Prediction Module
Loads datasets, applies preprocessing, loads trained models, and generates predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import data_preprocessing
import state_space

# Import dependencies with availability checks
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available")

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

def load_datasets():
    """Load and concatenate the example match datasets"""
    dataset_path = Path("dataset")
    
    # Load both datasets
    df1 = pd.read_csv(dataset_path / "example_matches.csv")
    df2 = pd.read_csv(dataset_path / "example_matches1.csv")
    
    # Concatenate datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

def load_best_models():
    """Load the best trained models for each role"""
    models_path = Path("models")
    roles = ['top', 'jungle', 'mid', 'adc', 'support']
    models = {}
    
    for role in roles:
        model_loaded = False
        
        # Try TensorFlow models first (.h5 extension)
        if TF_AVAILABLE:
            h5_file = models_path / f"best_{role}_model.h5"
            if h5_file.exists():
                try:
                    models[role] = load_model(str(h5_file), compile=False)
                    print(f"Loaded best_{role}_model.h5")
                    model_loaded = True
                except Exception as e:
                    print(f"Error loading {role} .h5 model: {e}")
        
        # Try joblib models (.pkl extension) as fallback
        if not model_loaded and JOBLIB_AVAILABLE:
            pkl_file = models_path / f"best_{role}_model.pkl"
            if pkl_file.exists():
                try:
                    models[role] = joblib.load(str(pkl_file))
                    print(f"Loaded best_{role}_model.pkl")
                    model_loaded = True
                except Exception as e:
                    print(f"Error loading {role} .pkl model: {e}")
        
        if not model_loaded:
            print(f"Warning: No model found for {role}")
    
    return models


def add_predictions(df, models):
    """Generate role-specific predictions and add to dataframe"""
    roles = ['top', 'jungle', 'mid', 'adc', 'support']
    
    # Initialize prediction columns
    for role in roles:
        df[f'{role}_prediction'] = np.nan
    
    for role in roles:
        if role in models and models[role] is not None:
            try:
                # Filter for role-specific players
                role_mask = df['Role'].str.lower() == role.lower()
                role_data = df[role_mask]
                
                if len(role_data) > 0:
                    # Process data once using the preprocessing pipeline
                    processor = data_preprocessing.DataProcessor()
                    df_engineered = processor.engineer_features(role_data)
                    role_datasets = processor.prepare_role_datasets(df_engineered)
                    
                    if role.upper() in role_datasets:
                        clean_role_data = role_datasets[role.upper()]
                        
                        # Apply scaling using preprocess_split
                        X_scaled, feature_names = processor.preprocess_split(clean_role_data)
                        
                        if X_scaled.shape[0] > 0:
                            # Generate predictions
                            predictions = models[role].predict(X_scaled, verbose=0)
                            predictions_flat = predictions.flatten() if predictions.ndim > 1 else predictions
                            
                            # Assign predictions to players with complete data
                            if len(predictions_flat) == len(clean_role_data):
                                df.loc[clean_role_data.index, f'{role}_prediction'] = predictions_flat
                                print(f"Added {role} predictions for {len(clean_role_data)} players")
                            else:
                                print(f"Shape mismatch for {role}: {len(predictions_flat)} vs {len(clean_role_data)}")
                        else:
                            print(f"No valid features for {role}")
                    else:
                        print(f"No {role} dataset found")
                else:
                    print(f"No {role} players found")
                    
            except Exception as e:
                print(f"Error making predictions for {role}: {e}")
        else:
            print(f"No model available for {role}")
    
    return df

def generate_team_performance_scores(df):
    """Generate team performance scores using individual player predictions"""
    print("Generating team performance scores...")
    
    # Group by game and team to calculate team scores
    team_scores = []
    
    for game_id in df['Game_ID'].unique():
        game_data = df[df['Game_ID'] == game_id]
        
        # Get blue and red team names
        blue_team = game_data['Blue_Team'].iloc[0]
        red_team = game_data['Red_Team'].iloc[0]
        winning_team = game_data['Winning_Team'].iloc[0]
        
        # In LoL, each game has exactly 10 players (5 per team)
        # We need to split them into blue and red teams
        # Assuming the first 5 players are blue team, next 5 are red team
        # Or we can use the Win column to determine team affiliation
        
        # Method: Use Win column - players with Win=1 are on winning team
        if len(game_data) == 10:  # Standard 5v5 match
            # Split players by their win status and team affiliation
            # Players on winning team have Win=1, losing team has Win=0
            winning_players = game_data[game_data['Win'] == 1]
            losing_players = game_data[game_data['Win'] == 0]
            
            # Determine which team won
            if winning_team == blue_team:
                blue_players = winning_players
                red_players = losing_players
            else:
                blue_players = losing_players
                red_players = winning_players
            
            # Sum individual player predictions for each team
            blue_score = blue_players['prediction'].sum()
            red_score = red_players['prediction'].sum()
            
            # Normalize scores to probabilities
            total_score = blue_score + red_score
            if total_score > 0:
                blue_prob = blue_score / total_score
                red_prob = red_score / total_score
            else:
                blue_prob = red_prob = 0.5
            
            team_scores.append({
                'Game_ID': game_id,
                'Blue_Team': blue_team,
                'Red_Team': red_team,
                'Blue_Score': blue_score,
                'Red_Score': red_score,
                'Blue_Win_Probability': blue_prob,
                'Red_Win_Probability': red_prob,
                'Actual_Winner': winning_team,
                'Blue_Players': len(blue_players),
                'Red_Players': len(red_players)
            })
        else:
            print(f"Warning: Game {game_id} has {len(game_data)} players (expected 10)")
    
    team_df = pd.DataFrame(team_scores)
    return team_df



def run_autoregressive_model(df_with_predictions, worlds_teams):
    """Run the autoregressive model on team performance data"""
    print("Running autoregressive team performance model...")
    
    # Create and fit the autoregressive model
    ar_model = state_space.TeamAutoregressiveModel(worlds_teams=worlds_teams)
    
    # Prepare team-level data from individual player predictions
    team_data = ar_model.prepare_team_data(df_with_predictions)
    print(f"Prepared team data: {len(team_data)} team-game records")
    
    # Fit autoregressive models for each team
    ar_model.fit()
    
    # Get model summary
    summary = ar_model.get_team_summary()
    print("\nAutoregressive Model Summary:")
    print(summary[['Team', 'Games_Played', 'Avg_Performance', 'Performance_Std']].head(10))
    
    # Test predictions for a few team matchups
    print("\nSample team matchup predictions:")
    teams = summary['Team'].tolist()[:6]  # Get first 6 teams
    
    comparison = set()  # use a set of frozensets for fast duplicate checking

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            pair = frozenset([teams[i], teams[j]])  # unordered, so {A,B} == {B,A}

            if pair not in comparison:
                team1, team2 = teams[i], teams[j]
                comparison.add(pair)  # remember we already predicted this pair

                prediction = ar_model.predict_match_outcome(team1, team2)
                print(f"  {team1} vs {team2}: "
                    f"{prediction['team1_win_probability']:.3f} - "
                    f"{prediction['team2_win_probability']:.3f}")
                    
    return ar_model, summary

def filter_worlds_teams(df, worlds_teams):
    """Filter dataset to only include Worlds teams"""
    # Define Worlds teams from swiss_stage.py
    
    print(f"Filtering for Worlds teams...")
    print(f"Original dataset shape: {df.shape}")
    
    # Filter for matches involving Worlds teams
    worlds_mask = (df['Blue_Team'].isin(worlds_teams)) | (df['Red_Team'].isin(worlds_teams))
    df_worlds = df[worlds_mask].copy()
    
    print(f"Worlds dataset shape: {df_worlds.shape}")
    print(f"Unique teams in Worlds data: {sorted(set(df_worlds['Blue_Team'].unique()) | set(df_worlds['Red_Team'].unique()))}")
    
    return df_worlds

def main():
    """Main pipeline: load data, preprocess, load models, and generate predictions"""
    # Load and combine datasets
    print("Loading datasets...")
    df = load_datasets()
    worlds_teams = [
        'Bilibili Gaming', 'Gen.G eSports', 'G2 Esports', 'FlyQuest', 
        'CTBC Flying Oyster', 'Anyone s Legend', 'Hanwha Life eSports', 
        'Movistar KOI', 'Vivo Keyd Stars', 'Team Secret Whales', 
        'Top Esports', 'Invictus Gaming', 'KT Rolster', 'Fnatic', 
        '100 Thieves', 'PSG Talon', 'T1'
    ]
    # Filter for Worlds teams only
    df = filter_worlds_teams(df, worlds_teams)
    
    # Apply feature engineering
    print("\nPreprocessing data...")
    processor = data_preprocessing.DataProcessor()
    df_processed = processor.engineer_features(df)
    print(f"Data after preprocessing: {df_processed.shape}")
    
    # Load trained models
    print("\nLoading best models...")
    models = load_best_models()
    
    # Generate predictions
    print("\nMaking predictions...")
    df_with_predictions = add_predictions(df_processed, models)
    
    # Create overall prediction column (take the non-NaN prediction for each player)
    pred_cols = [col for col in df_with_predictions.columns if 'prediction' in col]
    df_with_predictions['prediction'] = df_with_predictions[pred_cols].sum(axis=1, skipna=True)
    
    # Generate team performance scores
    print("\nGenerating team performance scores...")
    team_scores = generate_team_performance_scores(df_with_predictions)
    
    # Save results
    print("\nSaving results...")
    output_path = "dataset/matches_with_predictions.csv"
    df_with_predictions.to_csv(output_path, index=False)
    print(f"Saved player predictions to {output_path}")
    
    team_output_path = "dataset/team_performance_scores.csv"
    team_scores.to_csv(team_output_path, index=False)
    print(f"Saved team performance scores to {team_output_path}")
    
    # Display summary
    print(f"\nFinal dataset shape: {df_with_predictions.shape}")
    print(f"Team scores shape: {team_scores.shape}")
    
    if pred_cols:
        print(f"\nPlayer prediction summary:")
        print(df_with_predictions[pred_cols].describe())
    
    print(f"\nTeam performance summary:")
    print(team_scores[['Blue_Score', 'Red_Score', 'Blue_Win_Probability', 'Red_Win_Probability']].describe())
    
    # Calculate prediction accuracy
    team_scores['Predicted_Winner'] = team_scores.apply(
        lambda row: row['Blue_Team'] if row['Blue_Win_Probability'] > 0.5 else row['Red_Team'], axis=1
    )
    accuracy = (team_scores['Predicted_Winner'] == team_scores['Actual_Winner']).mean()
    print(f"\nTeam prediction accuracy: {accuracy:.3f}")
    
    # Run autoregressive model
    print("\n" + "="*50)
    print("RUNNING AUTOREGRESSIVE TEAM PERFORMANCE MODEL")
    print("="*50)
    
    try:
        ar_model, ar_summary = run_autoregressive_model(df_with_predictions, worlds_teams=worlds_teams)
        print("\nAutoregressive model completed successfully!")
        
        # Save autoregressive results
        ar_output = "dataset/autoregressive_results.csv"
        ar_summary.to_csv(ar_output, index=False)
        print(f"Saved autoregressive model summary to {ar_output}")
        
        return df_with_predictions, team_scores, ar_model, ar_summary
        
    except Exception as e:
        print(f"Error running autoregressive model: {e}")
        print("Continuing without autoregressive model...")
        return df_with_predictions, team_scores, None, None

if __name__ == "__main__":
    results = main()
    if len(results) == 4:
        df_final, team_scores, ar_model, ar_summary = results
        print(f"\nCompleted with autoregressive model: {ar_model is not None}")
        if ar_model is not None:
            print(f"Fitted models for {len(ar_model.team_models)} teams")
    else:
        df_final, team_scores = results
        print("\nCompleted without autoregressive model")

