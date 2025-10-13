import numpy as np
import pandas as pd
from itertools import combinations

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Skipping visualizations.")

def logit(p):
    """Convert probability to logit (log-odds)"""
    return np.log(p) - np.log(1 - p)

def softmax_win_probability(perf1, perf2):
    """
    Calculate win probability using softmax on performance scores.
    
    Args:
        perf1: Team 1 performance score (0-1)
        perf2: Team 2 performance score (0-1)
        
    Returns:
        team1_win_prob: Probability that team 1 wins
    """
    # Clip to avoid numerical issues
    perf1_clipped = np.clip(perf1, 0.001, 0.999)
    perf2_clipped = np.clip(perf2, 0.001, 0.999)
    
    # Convert to logits for softmax calculation
    logit1 = logit(perf1_clipped)
    logit2 = logit(perf2_clipped)
    
    # Apply softmax to get win probabilities
    max_logit = max(logit1, logit2)  # For numerical stability
    exp1 = np.exp(logit1 - max_logit)
    exp2 = np.exp(logit2 - max_logit)
    
    team1_win_prob = exp1 / (exp1 + exp2)
    return team1_win_prob

def load_team_predictions(csv_path="dataset/autoregressive_results.csv"):
    """
    Load team predictions from the autoregressive results CSV.
    
    Args:
        csv_path: Path to the autoregressive results CSV file
        
    Returns:
        dict: Dictionary mapping team names to their predicted performance
    """
    try:
        df = pd.read_csv(csv_path)
        team_predictions = {}
        
        for _, row in df.iterrows():
            team_name = row['Team']
            predicted_mean = row['Predicted_Mean']
            team_predictions[team_name] = predicted_mean
            
        return team_predictions
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Using sample data.")
        # Return sample data if file not found
        return {
            'FlyQuest': 0.785,
            'Gen.G eSports': 0.740,
            'Bilibili Gaming': 0.682,
            'G2 Esports': 0.675,
            'CTBC Flying Oyster': 0.701,
            'Anyone s Legend': 0.651
        }

def build_probability_matrix(team_predictions):
    """
    Build a probability matrix for all team matchups.
    
    Args:
        team_predictions: Dictionary mapping team names to predicted performance
        
    Returns:
        pd.DataFrame: Probability matrix where entry (i,j) is probability that team i beats team j
    """
    teams = list(team_predictions.keys())
    n_teams = len(teams)
    
    # Initialize matrix
    prob_matrix = np.zeros((n_teams, n_teams))
    
    # Fill the matrix
    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i == j:
                prob_matrix[i, j] = 0.5  # Team vs itself = 50%
            else:
                perf1 = team_predictions[team1]
                perf2 = team_predictions[team2]
                win_prob = softmax_win_probability(perf1, perf2)
                prob_matrix[i, j] = win_prob
    
    # Create DataFrame with team names as index and columns
    prob_df = pd.DataFrame(prob_matrix, index=teams, columns=teams)
    return prob_df

def generate_all_matchups(team_predictions):
    """
    Generate all possible matchups with win probabilities.
    
    Args:
        team_predictions: Dictionary mapping team names to predicted performance
        
    Returns:
        pd.DataFrame: All matchups with win probabilities
    """
    teams = list(team_predictions.keys())
    matchups = []
    
    for team1, team2 in combinations(teams, 2):
        perf1 = team_predictions[team1]
        perf2 = team_predictions[team2]
        
        win_prob_1 = softmax_win_probability(perf1, perf2)
        win_prob_2 = 1 - win_prob_1
        
        matchups.append({
            'Team_1': team1,
            'Team_2': team2,
            'Team_1_Performance': perf1,
            'Team_2_Performance': perf2,
            'Team_1_Win_Prob': win_prob_1,
            'Team_2_Win_Prob': win_prob_2,
            'Favorite': team1 if win_prob_1 > 0.5 else team2,
            'Favorite_Prob': max(win_prob_1, win_prob_2)
        })
    
    return pd.DataFrame(matchups)

def visualize_probability_matrix(prob_matrix, save_path="probability_matrix.png"):
    """
    Create a heatmap visualization of the probability matrix.
    
    Args:
        prob_matrix: DataFrame with probability matrix
        save_path: Path to save the visualization
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping visualization.")
        return
        
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(prob_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                center=0.5,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8})
    
    plt.title('Team vs Team Win Probability Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Opponent Team', fontsize=12)
    plt.ylabel('Team', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Probability matrix visualization saved to {save_path}")
    plt.show()

def analyze_team_strengths(team_predictions):
    """
    Analyze team strengths based on predicted performance.
    
    Args:
        team_predictions: Dictionary mapping team names to predicted performance
        
    Returns:
        pd.DataFrame: Team analysis with rankings and statistics
    """
    teams_analysis = []
    
    for team, performance in team_predictions.items():
        # Calculate expected win rate against all other teams
        win_probs = []
        for opponent, opp_performance in team_predictions.items():
            if team != opponent:
                win_prob = softmax_win_probability(performance, opp_performance)
                win_probs.append(win_prob)
        
        expected_win_rate = np.mean(win_probs)
        
        teams_analysis.append({
            'Team': team,
            'Predicted_Performance': performance,
            'Expected_Win_Rate': expected_win_rate,
            'Min_Win_Prob': min(win_probs),
            'Max_Win_Prob': max(win_probs),
            'Win_Prob_Std': np.std(win_probs)
        })
    
    df_analysis = pd.DataFrame(teams_analysis)
    df_analysis = df_analysis.sort_values('Expected_Win_Rate', ascending=False)
    df_analysis['Rank'] = range(1, len(df_analysis) + 1)
    
    return df_analysis[['Rank', 'Team', 'Predicted_Performance', 'Expected_Win_Rate', 
                      'Min_Win_Prob', 'Max_Win_Prob', 'Win_Prob_Std']]

def main():
    """
    Main function to build and analyze the probability matrix.
    """
    print("Building Probability Matrix from Team Predictions...")
    print("=" * 60)
    
    # Load team predictions
    team_predictions = load_team_predictions()
    print(f"Loaded predictions for {len(team_predictions)} teams")
    
    # Build probability matrix
    prob_matrix = build_probability_matrix(team_predictions)
    print(f"Built {prob_matrix.shape[0]}x{prob_matrix.shape[1]} probability matrix")
    
    # Save probability matrix
    prob_matrix.to_csv("dataset/probability_matrix.csv")
    print("Saved probability matrix to dataset/probability_matrix.csv")
    
    # Generate all matchups
    all_matchups = generate_all_matchups(team_predictions)
    all_matchups.to_csv("dataset/all_matchups.csv", index=False)
    print(f"Generated {len(all_matchups)} unique matchups")
    
    # Analyze team strengths
    team_analysis = analyze_team_strengths(team_predictions)
    team_analysis.to_csv("dataset/team_strength_analysis.csv", index=False)
    print("Team strength analysis completed")
    
    # Display results
    print("\n" + "=" * 60)
    print("PROBABILITY MATRIX (Top 6x6)")
    print("=" * 60)
    print(prob_matrix.iloc[:6, :6].round(3))
    
    print("\n" + "=" * 60)
    print("TEAM STRENGTH RANKINGS")
    print("=" * 60)
    print(team_analysis.round(3))
    
    print("\n" + "=" * 60)
    print("TOP 10 MOST COMPETITIVE MATCHUPS")
    print("=" * 60)
    competitive_matchups = all_matchups.copy()
    competitive_matchups['Competitiveness'] = 1 - abs(competitive_matchups['Team_1_Win_Prob'] - 0.5) * 2
    top_competitive = competitive_matchups.nlargest(10, 'Competitiveness')
    
    for _, match in top_competitive.iterrows():
        print(f"{match['Team_1']} vs {match['Team_2']}: "
              f"{match['Team_1_Win_Prob']:.1%} - {match['Team_2_Win_Prob']:.1%}")
    
    # Create visualization
    if PLOTTING_AVAILABLE:
        try:
            visualize_probability_matrix(prob_matrix)
        except Exception as e:
            print(f"Visualization failed: {e}")
            print("Continuing without visualization...")
    else:
        print("Skipping visualization (matplotlib not available)")
    
    print("\n" + "=" * 60)
    print("Probability matrix analysis complete!")
    print("Files saved:")
    print("- dataset/probability_matrix.csv")
    print("- dataset/all_matchups.csv") 
    print("- dataset/team_strength_analysis.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()