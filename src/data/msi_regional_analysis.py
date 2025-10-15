import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_msi_ewc_2025_results():
    """
    Analyze MSI 2025 + EWC 2025 results to generate empirical regional strength multipliers
    Combined tournament data for more accurate regional strength assessment
    """
    
    # MSI 2025 Results Data
    msi_results = [
        # Format: (Team1, Team2, Score1, Score2, Winner, Stage)
        ("Gen.G eSports", "T1", 3, 2, "Gen.G eSports", "FINALS"),
        ("T1", "Anyone s Legend", 3, 2, "T1", "ROUND4"),
        ("Anyone s Legend", "Bilibili Gaming", 3, 0, "Anyone s Legend", "ROUND3"),
        ("T1", "Gen.G eSports", 2, 3, "Gen.G eSports", "ROUND4"),
        ("FlyQuest", "Bilibili Gaming", 2, 3, "Bilibili Gaming", "ROUND2"),
        ("CTBC Flying Oyster", "Anyone s Legend", 1, 3, "Anyone s Legend", "ROUND2"),
        ("Bilibili Gaming", "T1", 0, 3, "T1", "ROUND2"),
        ("CTBC Flying Oyster", "Movistar KOI", 3, 1, "CTBC Flying Oyster", "ROUND1"),
        ("Anyone s Legend", "Gen.G eSports", 2, 3, "Gen.G eSports", "ROUND2"),
        ("FlyQuest", "G2 Esports", 3, 0, "FlyQuest", "ROUND1"),
        ("CTBC Flying Oyster", "T1", 2, 3, "T1", "ROUND1"),
        ("Movistar KOI", "Bilibili Gaming", 1, 3, "Bilibili Gaming", "ROUND1"),
        ("Anyone s Legend", "FlyQuest", 3, 1, "Anyone s Legend", "ROUND1"),
        ("Gen.G eSports", "G2 Esports", 3, 1, "Gen.G eSports", "ROUND1"),
        ("G2 Esports", "GAM Esports", 3, 2, "G2 Esports", "PLAYIN-DAY3"),
        ("Bilibili Gaming", "G2 Esports", 3, 0, "Bilibili Gaming", "PLAYIN-DAY2"),
        ("GAM Esports", "FURIA", 3, 2, "GAM Esports", "PLAYIN-DAY2"),
        ("Bilibili Gaming", "GAM Esports", 3, 0, "Bilibili Gaming", "PLAYIN-DAY1"),
        ("FURIA", "G2 Esports", 2, 3, "G2 Esports", "PLAYIN-DAY1"),
    ]
    
    # EWC 2025 Results Data (Esports World Cup) - Based on actual tournament results
    ewc_results = [
        # Format: (Team1, Team2, Score1, Score2, Winner, Stage)
        # Finals
        ("Gen.G eSports", "Anyone s Legend", 3, 2, "Gen.G eSports", "FINALS"),
        
        # 3rd Place
        ("G2 Esports", "T1", 0, 2, "T1", "3RD_PLACE"),
        
        # Semifinals  
        ("Gen.G eSports", "G2 Esports", 2, 1, "G2 Esports", "SEMIFINALS"),
        ("Anyone s Legend", "T1", 2, 0, "Anyone s Legend", "SEMIFINALS"),
        
        # Quarterfinals
        ("Gen.G eSports", "FlyQuest", 2, 0, "Gen.G eSports", "QUARTERFINALS"),
        ("T1", "Movistar KOI", 2, 1, "T1", "QUARTERFINALS"),
        ("Bilibili Gaming", "G2 Esports", 1, 2, "G2 Esports", "QUARTERFINALS"),
        ("Anyone s Legend", "Hanwha Life eSports", 2, 1, "Anyone s Legend", "QUARTERFINALS"),
        
        # Group Stage matches (key inter-regional results)
        ("FlyQuest", "FURIA", 1, 0, "FlyQuest", "GROUPSTAGE"),
        ("Movistar KOI", "CTBC Flying Oyster", 1, 0, "Movistar KOI", "GROUPSTAGE"),
        ("Cloud9", "FURIA", 0, 1, "FURIA", "GROUPSTAGE"),
        ("CTBC Flying Oyster", "GAM Esports", 0, 1, "GAM Esports", "GROUPSTAGE"),
        ("FlyQuest", "G2 Esports", 0, 1, "G2 Esports", "GROUPSTAGE"),
        ("Hanwha Life eSports", "Movistar KOI", 1, 0, "Hanwha Life eSports", "GROUPSTAGE"),
        ("FlyQuest", "Cloud9", 1, 0, "FlyQuest", "GROUPSTAGE"),
        ("G2 Esports", "FURIA", 1, 0, "G2 Esports", "GROUPSTAGE"),
        ("Movistar KOI", "GAM Esports", 1, 0, "Movistar KOI", "GROUPSTAGE"),
        ("Hanwha Life eSports", "CTBC Flying Oyster", 1, 0, "Hanwha Life eSports", "GROUPSTAGE"),
    ]
    
    # Combine MSI and EWC results for comprehensive analysis
    all_results = msi_results + ewc_results
    
    # Team to region mapping (comprehensive for MSI + EWC)
    team_regions = {
        # LCK teams (Korea)
        'Gen.G eSports': 'LCK',
        'T1': 'LCK',
        'Hanwha Life eSports': 'LCK',
        'KT Rolster': 'LCK',
        
        # LPL teams (China)
        'Bilibili Gaming': 'LPL',
        'Anyone s Legend': 'LPL',
        'Top Esports': 'LPL',
        
        # LEC teams (Europe)
        'G2 Esports': 'LEC',
        'Movistar KOI': 'LEC',
        'Fnatic': 'LEC',
        
        # LTA teams (Americas)
        'FlyQuest': 'LTA',
        'FURIA': 'LTA',
        '100 Thieves': 'LTA',
        'Vivo Keyd Stars': 'LTA',
        'Cloud9': 'LTA',
        
        # LCP teams (Taiwan/Vietnam)
        'CTBC Flying Oyster': 'LCP',
        'GAM Esports': 'LCP',
        'PSG Talon': 'LCP',
        'Team Secret Whales': 'LCP'
    }
    
    print("MSI 2025 + EWC 2025 Combined Regional Strength Analysis")
    print("=" * 65)
    
    # Track inter-regional matchups with tournament weighting
    regional_matchups = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0, 'games_won': 0, 'games_lost': 0, 'weighted_wins': 0, 'weighted_losses': 0}))
    
    # Tournament weights (MSI is more prestigious for regional strength assessment)
    tournament_weights = {
        'MSI': 1.2,  # MSI gets higher weight as it's the main international tournament
        'EWC': 1.0   # EWC gets standard weight
    }
    
    # Process both tournaments with weighting
    def process_tournament_results(results, tournament_name, weight):
        print(f"Processing {tournament_name} results (weight: {weight})...")
        for team1, team2, score1, score2, winner, stage in results:
            region1 = team_regions.get(team1, 'Unknown')
            region2 = team_regions.get(team2, 'Unknown')
            
            if region1 != 'Unknown' and region2 != 'Unknown' and region1 != region2:
                # Inter-regional match
                if winner == team1:
                    regional_matchups[region1][region2]['wins'] += 1
                    regional_matchups[region2][region1]['losses'] += 1
                    regional_matchups[region1][region2]['weighted_wins'] += weight
                    regional_matchups[region2][region1]['weighted_losses'] += weight
                else:
                    regional_matchups[region2][region1]['wins'] += 1
                    regional_matchups[region1][region2]['losses'] += 1
                    regional_matchups[region2][region1]['weighted_wins'] += weight
                    regional_matchups[region1][region2]['weighted_losses'] += weight
                
                # Track game scores
                regional_matchups[region1][region2]['games_won'] += score1
                regional_matchups[region1][region2]['games_lost'] += score2
                regional_matchups[region2][region1]['games_won'] += score2
                regional_matchups[region2][region1]['games_lost'] += score1
    
    # Process both tournaments
    process_tournament_results(msi_results, "MSI 2025", tournament_weights['MSI'])
    process_tournament_results(ewc_results, "EWC 2025", tournament_weights['EWC'])
    
    # Calculate regional win rates
    regional_stats = {}
    
    print("\nInter-Regional Head-to-Head Results:")
    print("-" * 45)
    
    for region1 in regional_matchups:
        total_wins = 0
        total_matches = 0
        total_weighted_wins = 0
        total_weighted_matches = 0
        total_games_won = 0
        total_games_played = 0
        
        print(f"\n{region1}:")
        for region2 in regional_matchups[region1]:
            wins = regional_matchups[region1][region2]['wins']
            losses = regional_matchups[region1][region2]['losses']
            weighted_wins = regional_matchups[region1][region2]['weighted_wins']
            weighted_losses = regional_matchups[region1][region2]['weighted_losses']
            games_won = regional_matchups[region1][region2]['games_won']
            games_lost = regional_matchups[region1][region2]['games_lost']
            
            matches = wins + losses
            weighted_matches = weighted_wins + weighted_losses
            if matches > 0:
                match_wr = wins / matches
                weighted_wr = weighted_wins / weighted_matches if weighted_matches > 0 else 0
                game_wr = games_won / (games_won + games_lost) if (games_won + games_lost) > 0 else 0
                print(f"  vs {region2}: {wins}-{losses} matches ({match_wr:.1%}), weighted: {weighted_wr:.1%}, {games_won}-{games_lost} games ({game_wr:.1%})")
                
                total_wins += wins
                total_matches += matches
                total_weighted_wins += weighted_wins
                total_weighted_matches += weighted_matches
                total_games_won += games_won
                total_games_played += games_won + games_lost
        
        if total_matches > 0:
            overall_match_wr = total_wins / total_matches
            overall_weighted_wr = total_weighted_wins / total_weighted_matches if total_weighted_matches > 0 else 0
            overall_game_wr = total_games_won / total_games_played if total_games_played > 0 else 0
            regional_stats[region1] = {
                'match_winrate': overall_match_wr,
                'weighted_winrate': overall_weighted_wr,
                'game_winrate': overall_game_wr,
                'total_matches': total_matches,
                'total_games': total_games_played
            }
            print(f"  Overall: {total_wins}-{total_matches-total_wins} matches ({overall_match_wr:.1%}), weighted: ({overall_weighted_wr:.1%}), {total_games_won}-{total_games_played-total_games_won} games ({overall_game_wr:.1%})")
    
    # Generate regional strength multipliers based on performance
    print(f"\n\nRegional Strength Analysis:")
    print("-" * 45)
    
    # Calculate strength based on both match and game win rates
    regional_strengths = {}
    
    # Base calculation using weighted results and game win rates
    base_strengths = {}
    for region, stats in regional_stats.items():
        # Combine weighted match results, game win rate, and tournament weighting
        # Weighted match results get highest priority, then game win rate
        combined_strength = (stats['weighted_winrate'] * 0.5) + (stats['game_winrate'] * 0.4) + (stats['match_winrate'] * 0.1)
        base_strengths[region] = combined_strength
    
    # Normalize to make strongest region = 1.0
    if base_strengths:
        max_strength = max(base_strengths.values())
        for region in base_strengths:
            regional_strengths[region] = base_strengths[region] / max_strength
    
    # Add regions that didn't have inter-regional matches (assign default values)
    all_regions = ['LCK', 'LPL', 'LEC', 'LTA', 'LCP']
    for region in all_regions:
        if region not in regional_strengths:
            regional_strengths[region] = 0.5  # Default for missing data
    
    # Display final regional strengths
    print(f"\nFinal Regional Strength Multipliers (MSI 2025 + EWC 2025 Combined):")
    print("-" * 70)
    
    sorted_regions = sorted(regional_strengths.items(), key=lambda x: x[1], reverse=True)
    for region, strength in sorted_regions:
        matches = regional_stats.get(region, {}).get('total_matches', 0)
        games = regional_stats.get(region, {}).get('total_games', 0)
        print(f"{region}: {strength:.3f} (based on {matches} matches, {games} games)")
    
    return regional_strengths

def create_msi_ewc_based_regional_adjustment():
    """
    Create a new regional adjustment file with MSI 2025 + EWC 2025 combined strengths
    """
    combined_strengths = analyze_msi_ewc_2025_results()
    
    print(f"\n\nCreating MSI-based regional adjustment...")
    
    # Create new regional adjustment content
    content = '''import pandas as pd
import numpy as np

def apply_regional_strength_adjustment(probability_matrix_path="dataset/probability_matrix.csv", 
                                     output_path="dataset/probability_matrix_msi_adjusted.csv"):
    """
    Apply regional strength adjustments based on MSI 2025 results.
    
    Regional strength hierarchy (based on MSI 2025 performance):
    1. LCK (Korea) - Dominated MSI 2025 (5-0 matches, 68.2% games)
    2. LPL (China) - Strong second (6-3 matches, 61.1% games)
    3. LCP (Taiwan/Vietnam) - Competitive performance including GAM
    4. LTA (Americas) - Struggled at MSI (1-4 matches, 45.5% games)
    5. LEC (Europe) - Worst major region (2-5 matches, 32.1% games)
    """
    
    # Load the original matrix
    prob_matrix = pd.read_csv(probability_matrix_path, index_col=0)
    
    # Define regional strength tiers based on MSI 2025 results
    regional_strength = {
'''
    
    # Add MSI-based strengths
    sorted_regions = sorted(combined_strengths.items(), key=lambda x: x[1], reverse=True)
    for region, strength in sorted_regions:
        content += f"        '{region}': {strength:.3f},    # MSI 2025 based\\n"
    
    content += '''    }
    
    # Team to region mapping
    team_regions = {
        # LPL teams (CN)
        'Bilibili Gaming': 'LPL',
        'Top Esports': 'LPL', 
        'Invictus Gaming': 'LPL',
        'Anyone s Legend': 'LPL',
        
        # LCK teams (KR) 
        'Gen.G eSports': 'LCK',
        'Hanwha Life eSports': 'LCK',
        'KT Rolster': 'LCK',
        'T1': 'LCK',
        
        # LEC teams (EUW)
        'G2 Esports': 'LEC',
        'Fnatic': 'LEC',
        'Movistar KOI': 'LEC',
        
        # LTA teams (Americas)
        'FlyQuest': 'LTA',
        '100 Thieves': 'LTA', 
        'Vivo Keyd Stars': 'LTA',
        
        # LCP teams (TW/VN)
        'PSG Talon': 'LCP',
        'CTBC Flying Oyster': 'LCP',
        'Team Secret Whales': 'LCP'
    }
    
    # Create adjusted matrix
    adjusted_matrix = prob_matrix.copy()
    
    print("Applying MSI 2025-based regional strength adjustments...")
    print("Regional strength multipliers:")
    for region, strength in regional_strength.items():
        print(f"  {region}: {strength:.3f}")
    
    # Apply adjustments
    for team1 in prob_matrix.index:
        for team2 in prob_matrix.columns:
            if team1 != team2:
                region1 = team_regions.get(team1, 'Unknown')
                region2 = team_regions.get(team2, 'Unknown')
                
                if region1 != 'Unknown' and region2 != 'Unknown':
                    # Get original probability
                    original_prob = prob_matrix.loc[team1, team2]
                    
                    # Calculate strength ratio
                    strength1 = regional_strength[region1]
                    strength2 = regional_strength[region2]
                    strength_ratio = strength1 / strength2
                    
                    # Apply extremely aggressive adjustment based on MSI data
                    if strength_ratio < 1.0:
                        # Weaker region vs stronger region - massive penalty
                        adjustment_factor = 0.25 + (strength_ratio * 0.5)  # Scale between 0.25-0.75
                        adjusted_prob = original_prob * adjustment_factor
                    elif strength_ratio > 1.0:
                        # Stronger region vs weaker region - significant boost
                        adjustment_factor = 1.0 + ((strength_ratio - 1.0) * 0.7)  # Scale between 1.0-1.7
                        adjusted_prob = original_prob * adjustment_factor
                    else:
                        # Same region strength
                        adjusted_prob = original_prob
                    
                    # Ensure probabilities stay within [0.05, 0.95] range for extreme realism
                    adjusted_prob = max(0.05, min(0.95, adjusted_prob))
                    
                    # Update matrix
                    adjusted_matrix.loc[team1, team2] = adjusted_prob
                    
                    # Ensure symmetry: P(A beats B) + P(B beats A) = 1
                    adjusted_matrix.loc[team2, team1] = 1.0 - adjusted_prob
    
    # Ensure diagonal is 0.5
    for team in adjusted_matrix.index:
        adjusted_matrix.loc[team, team] = 0.5
    
    # Save adjusted matrix
    adjusted_matrix.to_csv(output_path)
    
    # Show some key adjustments
    print(f"\\nKey adjustments for FlyQuest (LTA region):")
    for opponent in ['Gen.G eSports', 'Bilibili Gaming', 'Top Esports', 'KT Rolster']:
        if opponent in prob_matrix.columns:
            original = prob_matrix.loc['FlyQuest', opponent]
            adjusted = adjusted_matrix.loc['FlyQuest', opponent]
            change = adjusted - original
            print(f"  vs {opponent}: {original:.3f} → {adjusted:.3f} ({change:+.3f})")
    
    print(f"\\nMSI-adjusted probability matrix saved to: {output_path}")
    return adjusted_matrix

if __name__ == "__main__":
    adjusted_matrix = apply_regional_strength_adjustment()'''
    
    # Write the file
    with open('regional_adjustment_msi.py', 'w') as f:
        f.write(content)
    
    print(f"Created regional_adjustment_msi.py with MSI 2025-based regional strengths!")
    
    return combined_strengths

if __name__ == "__main__":
    combined_strengths = analyze_msi_ewc_2025_results()
    create_msi_ewc_based_regional_adjustment()