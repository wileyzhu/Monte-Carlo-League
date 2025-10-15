import pandas as pd
import numpy as np
from scipy.special import expit, logit

def adjust_prob(original_prob, strength_ratio, lambda_=0.8):
    return expit(logit(original_prob) + lambda_ * np.log(strength_ratio))

def apply_regional_strength_adjustment(probability_matrix_path="dataset/probability_matrix.csv", 
                                     output_path="dataset/probability_matrix_msi_adjusted.csv"):
    """
    Apply regional strength adjustments based on MSI 2025 + EWC 2025 combined results.
    
    Regional strength hierarchy (based on MSI 2025 + EWC 2025 combined performance):
    1. LCK (Korea) - Completely dominant (11-3 matches, 78.6% win rate, 80.0% weighted)
    2. LPL (China) - Strong but behind Korea (8-5 matches, 61.5% win rate, 62.2% weighted)
    3. LEC (Europe) - Competitive middle tier (8-8 matches, 50.0% win rate, 48.3% weighted)
    4. LCP (Taiwan/Vietnam) - Struggled significantly (2-7 matches, 22.2% win rate, 23.5% weighted)
    5. LTA (Americas) - Weakest region (1-7 matches, 12.5% win rate, 13.3% weighted)
    """
    
    # Load the original matrix
    prob_matrix = pd.read_csv(probability_matrix_path, index_col=0)
    
    # Define regional strength tiers based on MSI 2025 + EWC 2025 combined results
    regional_strength = {
        'LCK': 1.000,    # MSI 2025 + EWC 2025 combined
        'LPL': 0.821,    # MSI 2025 + EWC 2025 combined
        'LEC': 0.601,    # MSI 2025 + EWC 2025 combined
        'LCP': 0.393,    # MSI 2025 + EWC 2025 combined
        'LTA': 0.314,    # MSI 2025 + EWC 2025 combined
    }
    
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
                    
                    # Use logit-based adjustment for more mathematically sound probability transformation
                    adjusted_prob = adjust_prob(original_prob, strength_ratio, lambda_=1)
                
                    
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
    print(f"\nKey adjustments for FlyQuest (LTA region):")
    for opponent in ['Gen.G eSports', 'Bilibili Gaming', 'Top Esports', 'KT Rolster']:
        if opponent in prob_matrix.columns:
            original = prob_matrix.loc['FlyQuest', opponent]
            adjusted = adjusted_matrix.loc['FlyQuest', opponent]
            change = adjusted - original
            print(f"  vs {opponent}: {original:.3f} → {adjusted:.3f} ({change:+.3f})")
    
    print(f"\nMSI-adjusted probability matrix saved to: {output_path}")
    return adjusted_matrix

if __name__ == "__main__":
    adjusted_matrix = apply_regional_strength_adjustment()