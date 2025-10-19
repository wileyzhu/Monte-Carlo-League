import numpy as np
import random

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
        self.round_results = []  # Track detailed round results
        self.current_round = 0
        self.match_history = set()  # Track all matchups to avoid rematches

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
    
    def simulate_bo3_series(self, team_a, team_b):
        """
        Simulate a realistic BO3 series using single-game win probabilities.
        
        Returns:
            tuple: (winner, loser, final_score, games_list)
        """
        game_win_prob = self.win_probs[self.teams.index(team_a), self.teams.index(team_b)]
        
        games = []
        team_a_wins = 0
        team_b_wins = 0
        game_num = 1
        
        # Play games until one team reaches 2 wins
        while team_a_wins < 2 and team_b_wins < 2:
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
        if team_a_wins == 2:
            winner = team_a
            loser = team_b
            final_score = f'2-{team_b_wins}'
        else:
            winner = team_b
            loser = team_a
            final_score = f'2-{team_a_wins}'
        
        return winner, loser, final_score, games

    def play_round(self, pairs, best_of=None, round_name="Round"):
        """Play a round with given pairs of teams and track detailed results."""
        if best_of is None:
            best_of = self.best_of
            
        round_matches = []
        
        for a, b in pairs:
            # Record this matchup in history
            matchup = tuple(sorted([a, b]))
            self.match_history.add(matchup)
            # Get records before match
            a_record_before = f"{self.records[a][0]}-{self.records[a][1]}"
            b_record_before = f"{self.records[b][0]}-{self.records[b][1]}"
            
            if best_of == 3:
                # Use realistic BO3 simulation for elimination matches
                winner, loser, final_score, games = self.simulate_bo3_series(a, b)
                games_played = len(games)
            else:
                # BO1 match
                winner = self.match_series(a, b, best_of)
                loser = b if winner == a else a
                final_score = "1-0"
                games_played = 1
                games = [{'game_number': 1, 'winner': winner, 'score': '1-0'}]
            
            # Update records
            self.records[winner][0] += 1
            self.records[loser][1] += 1
            
            # Track match details
            match_result = {
                'team_a': a,
                'team_b': b,
                'winner': winner,
                'loser': loser,
                'score': final_score,
                'best_of': best_of,
                'games_played': games_played,
                'games': games,
                'team_a_record_before': a_record_before,
                'team_b_record_before': b_record_before,
                'team_a_record_after': f"{self.records[a][0]}-{self.records[a][1]}",
                'team_b_record_after': f"{self.records[b][0]}-{self.records[b][1]}"
            }
            round_matches.append(match_result)
        
        # Store round results
        round_result = {
            'round_name': f"{round_name} {self.current_round + 1}",
            'round_number': self.current_round + 1,
            'matches': round_matches
        }
        self.round_results.append(round_result)
        self.current_round += 1
    
    def play_mixed_round(self, matches_with_format, round_name="Swiss Round"):
        """Play a round with mixed BO1/BO3 matches simultaneously."""
        round_matches = []
        
        for a, b, best_of in matches_with_format:
            # Record this matchup in history
            matchup = tuple(sorted([a, b]))
            self.match_history.add(matchup)
            # Get records before match
            a_record_before = f"{self.records[a][0]}-{self.records[a][1]}"
            b_record_before = f"{self.records[b][0]}-{self.records[b][1]}"
            
            if best_of == 3:
                # Use realistic BO3 simulation for elimination matches
                winner, loser, final_score, games = self.simulate_bo3_series(a, b)
                games_played = len(games)
            else:
                # BO1 match
                winner = self.match_series(a, b, best_of)
                loser = b if winner == a else a
                final_score = "1-0"
                games_played = 1
                games = [{'game_number': 1, 'winner': winner, 'score': '1-0'}]
            
            # Update records
            self.records[winner][0] += 1
            self.records[loser][1] += 1
            
            # Track match details
            match_result = {
                'team_a': a,
                'team_b': b,
                'winner': winner,
                'loser': loser,
                'score': final_score,
                'best_of': best_of,
                'games_played': games_played,
                'games': games,
                'team_a_record_before': a_record_before,
                'team_b_record_before': b_record_before,
                'team_a_record_after': f"{self.records[a][0]}-{self.records[a][1]}",
                'team_b_record_after': f"{self.records[b][0]}-{self.records[b][1]}"
            }
            round_matches.append(match_result)
        
        # Store round results
        round_result = {
            'round_name': f"{round_name} {self.current_round + 1}",
            'round_number': self.current_round + 1,
            'matches': round_matches
        }
        self.round_results.append(round_result)
        self.current_round += 1

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
            top_region = self.team_regions.get(top_team, 'UNKNOWN')
            for low_team in low:
                if low_team not in used_low:
                    low_region = self.team_regions.get(low_team, 'UNKNOWN')
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

        while self.team_regions.get(last_mid, 'UNKNOWN') == self.team_regions.get(second_last_mid, 'UNKNOWN'):
            random.shuffle(available_mid)
            last_mid = available_mid[-1]
            second_last_mid = available_mid[-2]

        while len(available_mid) >= 2:
            team1 = available_mid.pop(0)
            team1_region = self.team_regions.get(team1, 'UNKNOWN')
            
            # Find a team from different region
            paired = False
            for i, team2 in enumerate(available_mid):
                team2_region = self.team_regions.get(team2, 'UNKNOWN')
                if team1_region != team2_region:
                    pairs.append((team1, team2))
                    available_mid.pop(i)
                    paired = True
                    break
        
        self.play_round(pairs, best_of=1, round_name="Opening Round")

    def swiss_round(self):
        """Subsequent Swiss rounds: group by record, pair within groups, avoiding rematches."""
        groups = {}
        for t, (w, l) in self.records.items():
            if w < 3 and l < 3:  # still alive
                groups.setdefault((w, l), []).append(t)

        # Collect ALL matches for this round
        all_matches = []
        has_elimination = False
        
        for group, tlist in groups.items():
            w, l = group
            random.shuffle(tlist)
            
            # Use pairing algorithm that avoids rematches
            paired = set()
            pairings = []
            
            for i, team_a in enumerate(tlist):
                if team_a in paired:
                    continue
                    
                # Try to find a valid opponent (not played before)
                for j in range(i + 1, len(tlist)):
                    team_b = tlist[j]
                    if team_b in paired:
                        continue
                    
                    matchup = tuple(sorted([team_a, team_b]))
                    if matchup not in self.match_history:
                        # Valid pairing found
                        pairings.append((team_a, team_b))
                        paired.add(team_a)
                        paired.add(team_b)
                        break
                else:
                    # No valid opponent found without rematch, pair with anyone remaining
                    for j in range(i + 1, len(tlist)):
                        team_b = tlist[j]
                        if team_b not in paired:
                            pairings.append((team_a, team_b))
                            paired.add(team_a)
                            paired.add(team_b)
                            break
            
            # Add pairings with appropriate format
            for a, b in pairings:
                # Decider matches (at 2–x) can be BO3
                if w == 2 or l == 2:
                    all_matches.append((a, b, 3))  # BO3
                    has_elimination = True
                else:
                    all_matches.append((a, b, 1))  # BO1
        
        # Play ALL matches in ONE single round (proper Swiss system)
        if all_matches:
            # All matches happen simultaneously in one round
            mixed_matches = []
            round_name = "Swiss Round"
            has_elimination = any(bo == 3 for _, _, bo in all_matches)
            
            if has_elimination:
                round_name = "Swiss Round (Mixed BO1/BO3)"
            
            # Process each match with its specific format
            for a, b, best_of in all_matches:
                mixed_matches.append((a, b, best_of))
            
            # Play all matches in one round call
            self.play_mixed_round(mixed_matches, round_name)

    def assign_seeds(self):
        top = [t for t, (w, l) in self.records.items() if l == 0]
        mid = [t for t, (w, l) in self.records.items() if l == 1]
        low = [t for t, (w, l) in self.records.items() if l == 2]

        return top, mid, low

    def run(self):
        """Run full Swiss stage for exactly 5 rounds (proper Swiss system)."""
        self.seeded_round1()
        
        # Run exactly 4 more rounds (5 total for 16 teams)
        for round_num in range(4):
            # Check if we have enough decided teams to stop early
            decided_teams = sum(1 for w, l in self.records.values() if w >= 3 or l >= 3)
            if decided_teams >= 14:  # If 14+ teams are decided, we can stop
                break
            self.swiss_round()
        
        # Determine qualified teams by ranking (top 8 by wins, then by losses)
        team_records = [(team, wins, losses) for team, (wins, losses) in self.records.items()]
        team_records.sort(key=lambda x: (-x[1], x[2]))  # Sort by wins desc, losses asc
        
        qualified = [team for team, _, _ in team_records[:8]]
        eliminated = [team for team, _, _ in team_records[8:]]
        
        return {
            'qualified': qualified,
            'eliminated': eliminated, 
            'records': self.records,
            'seeding': self.assign_seeds(),
            'round_results': self.round_results,
            'total_rounds': self.current_round
        }

