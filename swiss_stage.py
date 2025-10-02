import numpy as np
import random



from playin import Playin
class SwissTournament:
    def __init__(self, teams, seed_groups, team_regions, win_probs, best_of=1):
        """
        teams: list of team names
        seed_groups: dict {team: group_id}  (0=Top, 1=High, 2=Mid, 3=Low)
        win_probs: 2D numpy array, P[i,j] = prob team_i beats team_j
        best_of: default number of games for series (1 = BO1)
        """
        self.teams = teams
        self.seed_groups = seed_groups
        self.team_regions = team_regions
        self.win_probs = win_probs
        self.best_of = best_of
        self.records = {t: [0, 0] for t in teams}  # wins, losses

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

    def play_round(self, pairs, best_of=None):
        """Play a round with given pairs of teams."""
        for a, b in pairs:
            winner = self.match_series(a, b, best_of)
            loser = b if winner == a else a
            self.records[winner][0] += 1
            self.records[loser][1] += 1

    def seeded_round1(self):
        """Round 1: deterministic pairings by seed groups, avoiding same-region matchups."""
        top = [t for t, g in self.seed_groups.items() if g == 0]
        mid = [t for t, g in self.seed_groups.items() if g == 1]
        low = [t for t, g in self.seed_groups.items() if g == 2]
        
        random.shuffle(top)
        random.shuffle(mid)
        random.shuffle(low)
        
        pairs = []
        
        # Pair top seeds with low seeds, avoiding same regions
        used_low = set()
        for top_team in top:
            top_region = team_regions.get(top_team, 'UNKNOWN')
            for low_team in low:
                if low_team not in used_low:
                    low_region = team_regions.get(low_team, 'UNKNOWN')
                    if top_region != low_region:
                        pairs.append((top_team, low_team))
                        used_low.add(low_team)
                        break
            else:
                # If no different region found, pair with any remaining team
                for low_team in low:
                    if low_team not in used_low:
                        pairs.append((top_team, low_team))
                        used_low.add(low_team)
                        break
        
        # Pair remaining mid seeds, avoiding same regions
        available_mid = [t for t in mid]
        last_mid = available_mid[-1]
        second_last_mid = available_mid[-2]
        
        while team_regions.get(last_mid, 'UNKNOWN') == team_regions.get(second_last_mid, 'UNKNOWN'):
            random.shuffle(available_mid)

        while len(available_mid) >= 2:
            team1 = available_mid.pop(0)
            team1_region = team_regions.get(team1, 'UNKNOWN')
            
            # Find a team from different region
            paired = False
            for i, team2 in enumerate(available_mid):
                team2_region = team_regions.get(team2, 'UNKNOWN')
                if team1_region != team2_region:
                    pairs.append((team1, team2))
                    available_mid.pop(i)
                    paired = True
                    break
        
        self.play_round(pairs, best_of=1)

    def swiss_round(self):
        """Subsequent Swiss rounds: group by record, pair within groups."""
        groups = {}
        for t, (w, l) in self.records.items():
            if w < 3 and l < 3:  # still alive
                groups.setdefault((w, l), []).append(t)

        for group, tlist in groups.items():
            w, l = group
            random.shuffle(tlist)
            for i in range(0, len(tlist), 2):
                if i + 1 < len(tlist):
                    a, b = tlist[i], tlist[i+1]
                    # Decider matches (at 2–x) can be BO3
                    if w == 2 or l == 2:
                        self.play_round([(a, b)], best_of=3)
                    else:
                        self.play_round([(a, b)], best_of=1)

    def assign_seeds(self):
        top = [t for t, (w, l) in self.records.items() if l == 0]
        mid = [t for t, (w, l) in self.records.items() if l == 1]
        low = [t for t, (w, l) in self.records.items() if l == 2]

        return top, mid, low

    def run(self):
        """Run full Swiss stage until all teams are decided."""
        self.seeded_round1()
        while not all(w == 3 or l == 3 for w, l in self.records.values()):
            self.swiss_round()

        qualified = [t for t, (w, _) in self.records.items() if w == 3]
        eliminated = [t for t, (_, l) in self.records.items() if l == 3]
        return qualified, eliminated, self.records, self.assign_seeds()


# ----- Example usage -----
if __name__ == "__main__":
    teams1 = ['BLG', 'Geng', 'G2', 'Flyquest', 'CFO', 'AL', 'HLE', 'MKOI', 'VKS',
            'TSW', 'TES', 'IG', 'KT', 'FNC', '100T','PSG']
    teams2 = ['BLG', 'Geng', 'G2', 'Flyquest', 'CFO', 'AL', 'HLE', 'MKOI', 'VKS',
            'TSW', 'KT', 'T1', 'TES', 'FNC', '100T','PSG']

    team_regions = {
    # LPL teams
    'BLG': 'LPL',
    'TES': 'LPL',
    'IG': 'LPL',
    
    # LCK teams
    'Geng': 'LCK',
    'HLE': 'LCK',
    'KT': 'LCK',
    'T1': 'LCK',
    
    # LEC teams
    'G2': 'LEC',
    'FNC': 'LEC',
    
    # LTA teams
    'Flyquest': 'LTA',
    '100T': 'LTA',
    
    # PCS teams
    'PSG': 'PCS',
    'CFO': 'PCS',
    
    # Other regions/wildcards
    'AL': 'LPL',
    'MKOI': 'LEC',  # Assuming LEC for now
    'VKS': 'LTA',  # Assuming LTA for now
    'TSW': 'PCS'   # Assuming PCS for now
}

    playin_teams = ['IG', 'T1']
    win_probs = np.random.rand(2,2)

    playin = Playin(playin_teams, win_probs, best_of=5)
    playin_team = playin.run()
    if playin_team == 'IG':
        teams = teams1
    else:
        teams = teams2

    seed_groups = {t: (0 if i < 4 else 1 if i < 8 else 2 if i < 12 else 3)
               for i, t in enumerate(teams)}
    np.random.seed(42)
    win_probs = np.random.rand(16, 16)
    np.fill_diagonal(win_probs, 0.5)

    # Run one Swiss tournament
    swiss = SwissTournament(teams, seed_groups, team_regions, win_probs)
    qualified, eliminated, records, seeding = swiss.run()


    print("Qualified:", qualified)
    print("Eliminated:", eliminated)
    print("Records:", records)
    print("Seeding:", seeding)