import pandas as pd
import numpy as np

def apply_regional_strength_adjustment(probability_matrix_path="dataset/probability_matrix.csv", 
                                     output_path="dataset/probability_matrix_msi_adjusted.csv"):
    """
    Apply regional strength adjustments based on MSI 2025 results.
    
    Regional strength hierarchy (based on MSI 2025 performance):
    1. LCK (Korea) - Dominated MSI 2025 (5-0 matches, 68.2% games)
    2. LPL (China) - Strong second (6-3 matches, 61.1% games)
    3. PCS (Taiwan/Vietnam) - Competitive performance including GAM
    4. LTA (Americas) - Struggled at MSI (1-4 matches, 45.5% games)
    5. LEC (Europe) - Worst major region (2-5 matches, 32.1% games)
    """
    
    # Load the original matrix
    prob_matrix = pd.read_csv(probability_matrix_path, index_col=0)
    
    # Define regional strength tiers based on MSI 2025 results
    regional_strength = {
        'LCK': 1.000,    # MSI 2025 based\n        'LPL': 0.821,    # MSI 2024 based\n        'LEC': 0.601,    # MSI 2024 based\n        'PCS': 0.393,    # MSI 2024 based\n        'LTA': 0.314,    # MSI 2024 based\n    }
    
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
        
        # PCS teams (TW/VN)
        'PSG Talon': 'PCS',
        'CTBC Flying Oyster': 'PCS',
        'Team Secret Whales': 'PCS'
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