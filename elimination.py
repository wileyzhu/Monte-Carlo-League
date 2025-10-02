import numpy as np
import random
from swiss_stage import SwissTournament

class Elimination(SwissTournament):
    def __init__(self, teams, seed_groups, win_probs, best_of=5):
        super().__init__(teams, seed_groups, win_probs, best_of)


    def quarterfinals(self, seed_groups):
        high, mid ,low = seed_groups[0], seed_groups[1], seed_groups[2]

        random.shuffle(high)
        random.shuffle(mid)
        random.shuffle(low)

        matches = []
        matches.append((high[0], low[0]))
        matches.append((high[1], low[1]))
        matches.append((mid[0], low[2]))
        matches.append((mid[1], mid[2]))

        top_half = high[0] + low[0] + mid[0] + low[2]
        bot_half = high[1] + low[1] + mid[1] + mid[2]
        
        winners = []
        for team_a, team_b in matches:
            winners.append(self.match_series(team_a, team_b, best_of=5))
        
        top_winners = []
        bot_winners = []
        for winner in winners:
            if winner in top_half:
                top_winners.append(winner)
            else:
                bot_winners.append(winner)
        return top_winners, bot_winners

    def semifinals(self, top_winners, bot_winners):
        top_winner =  self.match_series(top_winners[0], top_winners[1], best_of=5)
        bot_winner = self.match_series(bot_winners[0], bot_winners[1], best_of=5)

        return top_winner, bot_winner

    def final(self, top_winner, bot_winner):
        return self.match_series(top_winner, bot_winner, best_of=5)

    def run(self):
        semi1, semi2 = self.quarterfinals(self.seed_groups)
        final1, final2 = self.semifinals(semi1, semi2)
        winner = self.final(final1, final2)
        return (semi1, semi2), (final1, final2), winner

if __name__ == "__main__":
    teams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    seed_groups = [['A', 'B'], ['C', 'D', 'E'], ['F', 'G', 'H']]
    np.random.seed(42)
    win_probs = np.random.rand(8, 8)
    np.fill_diagonal(win_probs, 0.5)
    tournament = Elimination(teams, seed_groups, win_probs)
    records = tournament.run()

    print(records)



    
            
        


