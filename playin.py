import numpy as np

class Playin():
    def  __init__(self, teams, win_probs, best_of = 5):
        self.teams = teams
        self.win_probs = win_probs
        self.best_of = best_of
        self.playin_result = None
        
    def match(self, team_a, team_b):
        """Simulate a single map between team_a and team_b."""
        p = self.win_probs[self.teams.index(team_a), self.teams.index(team_b)]
        return team_a if np.random.rand() < p else team_b

    def match_series(self, team_a, team_b, best_of=None):
        """Simulate a best-of-N series."""
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

    def win(self, teams, best_of = 5):
        team_a = teams[0]
        team_b = teams[1]
        return self.match_series(team_a, team_b, best_of)

    def run(self):
        self.playin_result = self.win(self.teams, self.best_of)
        return self.playin_result

if __name__ == '__main__':
    teams = ['IG', 'T1']
    win_probs = np.random.rand(2,2)
    playin = Playin(teams, win_probs, best_of=5)
    result = playin.run()
    print("Play-in result:", result)