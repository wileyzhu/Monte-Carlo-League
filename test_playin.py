#!/usr/bin/env python3
"""
Test script for play-in simulation functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.tournament.playin import Playin
import numpy as np

def test_playin_simulation():
    """Test the play-in simulation with T1 vs Invictus Gaming"""
    
    print("=== Play-in Simulation Test ===")
    print()
    
    # Teams
    teams = ['T1', 'Invictus Gaming']
    
    # Win probabilities based on team strength analysis
    # T1: 0.6646 predicted performance vs IG: 0.5895
    t1_win_prob = 0.6646 / (0.6646 + 0.5895)  # Normalize: ~0.53
    win_probs = np.array([[0.5, t1_win_prob], [1-t1_win_prob, 0.5]])
    
    print(f"Teams: {teams}")
    print(f"T1 win probability: {t1_win_prob:.3f} ({t1_win_prob*100:.1f}%)")
    print(f"IG win probability: {1-t1_win_prob:.3f} ({(1-t1_win_prob)*100:.1f}%)")
    print()
    
    # Create play-in instance
    playin = Playin(teams, win_probs, best_of=5)
    
    # Test single detailed simulation
    print("--- Single Detailed Simulation ---")
    result = playin.run_detailed_simulation()
    
    print(f"Winner: {result['winner']}")
    print(f"Final Score: {result['final_score']}")
    print(f"Series Length: {result['series_length']} games")
    print(f"Upset: {'Yes' if result['upset'] else 'No'}")
    print()
    
    print("Game-by-game results:")
    for game in result['games']:
        print(f"  Game {game['game_number']}: {game['winner']} (Score: {game['score']})")
    print()
    
    # Test multiple simulations
    print("--- Multiple Simulations (100 runs) ---")
    multi_result = playin.run_multiple_simulations(100)
    
    print(f"T1 wins: {multi_result['teams']['team_a']['wins']}/100 ({multi_result['teams']['team_a']['win_percentage']})")
    print(f"IG wins: {multi_result['teams']['team_b']['wins']}/100 ({multi_result['teams']['team_b']['win_percentage']})")
    print(f"Most likely winner: {multi_result['most_likely_winner']}")
    print(f"Confidence: {multi_result['confidence']*100:.1f}%")
    print()
    
    print("Recent simulation results:")
    for i, sim in enumerate(multi_result['all_results'][-5:], 1):
        upset_text = " (UPSET!)" if sim['upset'] else ""
        print(f"  Sim {i}: {sim['winner']} wins {sim['final_score']}{upset_text}")
    
    print()
    print("=== Test Complete ===")

if __name__ == '__main__':
    test_playin_simulation()