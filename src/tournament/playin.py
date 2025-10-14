import numpy as np
import random

class Playin():
    def  __init__(self, teams, win_probs, best_of = 5):
        self.teams = teams
        self.win_probs = win_probs
        self.best_of = best_of
        self.playin_result = None
        self.series_details = None
        
    def match(self, team_a, team_b):
        """Simulate a single map between team_a and team_b."""
        p = self.win_probs[self.teams.index(team_a), self.teams.index(team_b)]
        return team_a if np.random.rand() < p else team_b

    def simulate_bo5_series(self, team_a, team_b, game_win_prob):
        """
        Simulate a realistic BO5 series using single-game win probabilities.
        
        Args:
            team_a: First team
            team_b: Second team  
            game_win_prob: Probability that team_a wins a single game
            
        Returns:
            tuple: (winner, loser, final_score, games_list)
        """
        games = []
        team_a_wins = 0
        team_b_wins = 0
        game_num = 1
        
        # Play games until one team reaches 3 wins
        while team_a_wins < 3 and team_b_wins < 3:
            # Simulate single game based on win probability
            if random.random() < game_win_prob:
                game_winner = team_a
                team_a_wins += 1
            else:
                game_winner = team_b
                team_b_wins += 1
            
            games.append({
                'game_number': game_num,
                'winner': game_winner,
                'score': f'{team_a_wins}-{team_b_wins}'
            })
            
            game_num += 1
        
        # Determine series winner and final score
        if team_a_wins == 3:
            winner = team_a
            loser = team_b
            final_score = f'3-{team_b_wins}'
        else:
            winner = team_b
            loser = team_a
            final_score = f'3-{team_a_wins}'
        
        return winner, loser, final_score, games

    def match_series_realistic(self, team_a, team_b, best_of=None):
        """Simulate a realistic best-of-N series using single-game win probabilities."""
        if best_of is None:
            best_of = self.best_of
            
        # Get single-game win probability for team_a vs team_b
        game_win_prob = self.win_probs[self.teams.index(team_a), self.teams.index(team_b)]
        
        # Simulate the BO5 series game by game
        winner, loser, final_score, games = self.simulate_bo5_series(team_a, team_b, game_win_prob)
        
        # Store series details
        self.series_details = {
            'winner': winner,
            'loser': loser,
            'final_score': final_score,
            'games': games,
            'series_length': len(games),
            'win_probability': game_win_prob,  # Keep this key for compatibility
            'game_win_probability': game_win_prob,
            'series_win_probability': self.calculate_series_win_probability(game_win_prob)
        }
        
        return winner
    
    def calculate_series_win_probability(self, p):
        """
        Calculate the probability of winning a BO5 series given single-game win probability.
        
        Args:
            p: Single-game win probability
            
        Returns:
            Probability of winning the BO5 series
        """
        # BO5 series win probability calculation
        # Need to win 3 games before opponent wins 3
        # Possible winning scenarios: 3-0, 3-1, 3-2
        
        # 3-0: p^3
        prob_3_0 = p**3
        
        # 3-1: Choose 3 wins from first 4 games, then win game 4
        # C(3,3) * p^3 * (1-p)^0 * p = p^4
        # But we need exactly 3 wins in first 3 games of a 4-game sequence
        # Actually: ways to get 3-1 = C(3,2) * p^3 * (1-p)^1 = 3 * p^3 * (1-p)
        prob_3_1 = 3 * (p**3) * (1-p)
        
        # 3-2: Choose 3 wins from first 5 games, with game 5 being a win
        # C(4,2) * p^3 * (1-p)^2 = 6 * p^3 * (1-p)^2
        prob_3_2 = 6 * (p**3) * ((1-p)**2)
        
        return prob_3_0 + prob_3_1 + prob_3_2

    def match_series(self, team_a, team_b, best_of=None):
        """Simulate a best-of-N series (legacy method for compatibility)."""
        if best_of is None:
            best_of = self.best_of
        needed = best_of // 2 + 1
        wins_a, wins_b = 0, 0

        while wins_a < needed and wins_b < needed:
            winner = self.match(team_a, team_b)
            if winner == team_a:
                wins_a += 1
            else:
                wins_b += 1

        return team_a if wins_a > wins_b else team_b

    def win(self, teams, best_of = 5, realistic=True):
        team_a = teams[0]
        team_b = teams[1]
        if realistic:
            return self.match_series_realistic(team_a, team_b, best_of)
        else:
            return self.match_series(team_a, team_b, best_of)

    def run(self, realistic=True):
        """Run playin with realistic score distributions"""
        self.playin_result = self.win(self.teams, self.best_of, realistic)
        return self.playin_result
    
    def get_series_details(self):
        """Get detailed series information including score and games"""
        return self.series_details

