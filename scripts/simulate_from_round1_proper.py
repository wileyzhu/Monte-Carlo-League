#!/usr/bin/env python3
"""
Proper simulation from Round 1 results
Actually continues Swiss stage from current state
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import random
from src.tournament.swiss_stage import SwissTournament

def simulate_from_round1_proper(num_simulations=500):
    """
    Properly simulate Swiss stage from actual Round 1 results
    """
    
    # Load probability matrix
    prob_matrix = pd.read_csv('dataset/probability_matrix_msi_adjusted.csv', index_col=0)
    
    # Swiss teams (T1 won play-in)
    swiss_teams = [
        'Bilibili Gaming', 'Gen.G eSports', 'G2 Esports', 'FlyQuest',
        'CTBC Flying Oyster', 'Anyone s Legend', 'Hanwha Life eSports',
        'Movistar KOI', 'Vivo Keyd Stars', 'Team Secret Whales',
        'KT Rolster', 'T1', 'Top Esports', 'Fnatic',
        '100 Thieves', 'PSG Talon'
    ]
    
    # Team regions
    team_regions = {
        'Bilibili Gaming': 'LPL', 'Top Esports': 'LPL', 'Anyone s Legend': 'LPL',
        'Gen.G eSports': 'LCK', 'Hanwha Life eSports': 'LCK', 'KT Rolster': 'LCK', 'T1': 'LCK',
        'G2 Esports': 'LEC', 'Fnatic': 'LEC', 'Movistar KOI': 'LEC',
        'FlyQuest': 'LTA', '100 Thieves': 'LTA', 'Vivo Keyd Stars': 'LTA',
        'PSG Talon': 'LCP', 'CTBC Flying Oyster': 'LCP', 'Team Secret Whales': 'LCP'
    }
    
    # ACTUAL Round 1 results
    round1_records = {
        # 1-0 teams (Round 1 winners)
        'Team Secret Whales': [1, 0],    # beat Vivo Keyd Stars
        'CTBC Flying Oyster': [1, 0],    # beat Fnatic
        'KT Rolster': [1, 0],            # beat Movistar KOI
        '100 Thieves': [1, 0],           # beat Bilibili Gaming
        'T1': [1, 0],                    # beat FlyQuest
        'Anyone s Legend': [1, 0],       # beat Hanwha Life eSports
        'Top Esports': [1, 0],           # beat G2 Esports
        'Gen.G eSports': [1, 0],         # beat PSG Talon
        
        # 0-1 teams (Round 1 losers)
        'Vivo Keyd Stars': [0, 1],       # lost to Team Secret Whales
        'Fnatic': [0, 1],                # lost to CTBC Flying Oyster
        'Movistar KOI': [0, 1],          # lost to KT Rolster
        'Bilibili Gaming': [0, 1],       # lost to 100 Thieves
        'FlyQuest': [0, 1],              # lost to T1
        'Hanwha Life eSports': [0, 1],   # lost to Anyone s Legend
        'G2 Esports': [0, 1],            # lost to Top Esports
        'PSG Talon': [0, 1]              # lost to Gen.G eSports
    }
    
    print("=" * 60)
    print("PROPER SWISS SIMULATION FROM ROUND 1 RESULTS")
    print("=" * 60)
    print(f"Running {num_simulations} simulations from actual Round 1 state...")
    
    print("\nRound 1 Results:")
    print("1-0 Teams:", [team for team, record in round1_records.items() if record == [1, 0]])
    print("0-1 Teams:", [team for team, record in round1_records.items() if record == [0, 1]])
    
    # Create probability matrix for Swiss teams
    n_teams = len(swiss_teams)
    win_probs = np.zeros((n_teams, n_teams))
    
    for i, team1 in enumerate(swiss_teams):
        for j, team2 in enumerate(swiss_teams):
            if i == j:
                win_probs[i, j] = 0.5
            else:
                try:
                    win_probs[i, j] = prob_matrix.loc[team1, team2]
                except KeyError:
                    win_probs[i, j] = 0.5
    
    # Run simulations
    qualification_counts = {}
    all_final_records = {}
    
    for sim in range(num_simulations):
        if (sim + 1) % 100 == 0:
            print(f"Completed {sim + 1}/{num_simulations} simulations")
        
        # Start with Round 1 records
        current_records = {team: record.copy() for team, record in round1_records.items()}
        
        # Continue Swiss stage from Round 2
        # Simulate remaining rounds until all teams have 3 wins or 3 losses
        round_num = 2
        
        while not all(wins >= 3 or losses >= 3 for wins, losses in current_records.values()):
            # Determine matchups for this round (simplified Swiss pairing)
            teams_by_record = {}
            for team, (wins, losses) in current_records.items():
                if wins < 3 and losses < 3:  # Team still active
                    record_key = f"{wins}-{losses}"
                    if record_key not in teams_by_record:
                        teams_by_record[record_key] = []
                    teams_by_record[record_key].append(team)
            
            # Pair teams within same record groups
            round_matches = []
            for record_group, teams in teams_by_record.items():
                random.shuffle(teams)
                for i in range(0, len(teams) - 1, 2):
                    if i + 1 < len(teams):
                        round_matches.append((teams[i], teams[i + 1]))
            
            # Simulate matches
            for team1, team2 in round_matches:
                try:
                    win_prob = prob_matrix.loc[team1, team2]
                except KeyError:
                    win_prob = 0.5
                
                if random.random() < win_prob:
                    winner, loser = team1, team2
                else:
                    winner, loser = team2, team1
                
                # Update records
                current_records[winner][0] += 1  # Add win
                current_records[loser][1] += 1   # Add loss
            
            round_num += 1
            if round_num > 10:  # Safety break
                break
        
        # Determine qualified teams (3+ wins)
        qualified = [team for team, (wins, losses) in current_records.items() if wins >= 3]
        
        # Count qualifications
        for team in qualified:
            qualification_counts[team] = qualification_counts.get(team, 0) + 1
        
        # Track final records
        for team, (wins, losses) in current_records.items():
            record_key = f"{wins}-{losses}"
            if record_key not in all_final_records:
                all_final_records[record_key] = 0
            all_final_records[record_key] += 1
    
    # Calculate results
    qual_stats = []
    for team, count in qualification_counts.items():
        percentage = (count / num_simulations) * 100
        region = team_regions.get(team, 'Unknown')
        qual_stats.append({
            'team': team,
            'region': region,
            'qualifications': count,
            'percentage': percentage
        })
    
    qual_stats.sort(key=lambda x: x['percentage'], reverse=True)
    
    # Display results
    print("\n" + "=" * 60)
    print("QUALIFICATION PROBABILITIES (PROPER SIMULATION)")
    print("=" * 60)
    
    print(f"\nQualification chances (based on {num_simulations} simulations):")
    print("-" * 60)
    
    for i, stat in enumerate(qual_stats, 1):
        record_status = "1-0" if stat['team'] in [t for t, r in round1_records.items() if r == [1, 0]] else "0-1"
        print(f"{i:2d}. {stat['team']:<25} {stat['percentage']:5.1f}% ({record_status}) ({stat['region']})")
    
    print(f"\nRecord Distribution:")
    print("-" * 30)
    for record, count in sorted(all_final_records.items()):
        percentage = (count / num_simulations) * 100
        print(f"{record}: {count:4d} ({percentage:5.1f}%)")
    
    return qual_stats

if __name__ == "__main__":
    results = simulate_from_round1_proper(500)