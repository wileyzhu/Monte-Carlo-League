import pandas as pd
import numpy as np
import random
from .playin import Playin
from .swiss_stage import SwissTournament
from .elimination import Elimination

class WorldsTournament:
    """
    Complete Worlds tournament simulation using existing playin, swiss_stage, and elimination modules
    with the probability matrix generated from team predictions.
    """
    
    def __init__(self, probability_matrix_path="dataset/probability_matrix_msi_adjusted.csv"):
        """
        Initialize tournament with probability matrix.
        
        Args:
            probability_matrix_path: Path to the probability matrix CSV
        """
        self.prob_matrix_df = pd.read_csv(probability_matrix_path, index_col=0)
        self.all_teams = list(self.prob_matrix_df.index)
        
        # Team regions mapping
        self.team_regions = {
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
            'Vivo Keyd Stars': 'LTA',  # LAT region
            
            # PCS teams (TW/VN)
            'PSG Talon': 'PCS',
            'CTBC Flying Oyster': 'PCS',
            'Team Secret Whales': 'PCS'   # VN region
        }
        
    def get_win_probability(self, team1, team2):
        """Get win probability for team1 vs team2 from existing matrix"""
        try:
            return self.prob_matrix_df.loc[team1, team2]
        except KeyError:
            return 0.5  # Default 50-50 if team not found
    
    def setup_tournament_teams(self, num_teams=16):
        """
        Setup teams for the tournament based on strength rankings.
        
        Args:
            num_teams: Total number of teams for the tournament
            
        Returns:
            dict with play_in_teams, direct_swiss_teams, and team assignments
        """
        # Use all teams from probability matrix
        available_teams = [team for team in self.all_teams if team in self.team_regions]
        
        # Calculate team strengths from probability matrix
        team_strengths = []
        for team in available_teams:
            win_rates = []
            for opponent in available_teams:
                if team != opponent:
                    win_rates.append(self.get_win_probability(team, opponent))
            avg_win_rate = np.mean(win_rates) if win_rates else 0.5
            team_strengths.append((team, avg_win_rate))
        
        # Sort by strength
        team_strengths.sort(key=lambda x: x[1], reverse=True)
        
        # Split teams: top teams go directly to Swiss, others to Play-In
        num_teams = len(available_teams)
        if num_teams >= 16:
            direct_swiss_teams = [team[0] for team in team_strengths[:12]]
            play_in_teams = [team[0] for team in team_strengths[12:16]]
        else:
            # Adjust for fewer teams
            direct_swiss_teams = [team[0] for team in team_strengths[:max(8, num_teams-4)]]
            play_in_teams = [team[0] for team in team_strengths[len(direct_swiss_teams):]]
        
        return {
            'play_in_teams': play_in_teams,
            'direct_swiss_teams': direct_swiss_teams,
            'all_selected_teams': [team[0] for team in team_strengths],
            'team_strengths': team_strengths
        }
    
    def run_play_in(self, play_in_teams):
        """
        Run Play-In stage to determine advancing teams using realistic BO5 simulation.
        
        Args:
            play_in_teams: List of teams in Play-In
            
        Returns:
            List of teams advancing to Swiss Stage
        """
        print("=" * 60)
        print("PLAY-IN STAGE")
        print("=" * 60)
        print(f"Play-In teams: {play_in_teams}")
        
        if len(play_in_teams) < 2:
            print("Not enough teams for Play-In. All advance.")
            return play_in_teams
        
        # For 2 teams (T1 vs Invictus Gaming), use realistic playin simulation
        if len(play_in_teams) == 2:
            team1, team2 = play_in_teams
            
            # Create win probability matrix
            playin_win_probs = np.zeros((2, 2))
            for i, t1 in enumerate(play_in_teams):
                for j, t2 in enumerate(play_in_teams):
                    if i == j:
                        playin_win_probs[i, j] = 0.5
                    else:
                        playin_win_probs[i, j] = self.get_win_probability(t1, t2)
            
            # Run realistic playin with score distributions
            playin = Playin(play_in_teams, playin_win_probs, best_of=5)
            winner = playin.run(realistic=True)
            series_details = playin.get_series_details()
            
            print(f"\nPlay-In Final (BO5)")
            print("-" * 30)
            print(f"  {team1} vs {team2}")
            print(f"  Winner: {winner}")
            print(f"  Final Score: {series_details['final_score']}")
            print(f"  Series Length: {series_details['series_length']} games")
            
            return [winner]
        
        # For more teams, run pairwise eliminations (legacy method)
        advancing_teams = []
        remaining_teams = play_in_teams.copy()
        
        round_num = 1
        while len(remaining_teams) > 2:
            print(f"\nPlay-In Round {round_num}")
            print("-" * 30)
            
            next_round = []
            random.shuffle(remaining_teams)
            
            # Pair teams and simulate matches
            for i in range(0, len(remaining_teams), 2):
                if i + 1 < len(remaining_teams):
                    team1 = remaining_teams[i]
                    team2 = remaining_teams[i + 1]
                    
                    # Get win probability from existing matrix
                    win_prob = self.get_win_probability(team1, team2)
                    
                    # Simulate BO5 match
                    if random.random() < win_prob:
                        winner = team1
                    else:
                        winner = team2
                    
                    print(f"  {team1} vs {team2}: {winner} advances")
                    next_round.append(winner)
                else:
                    # Odd number of teams, bye
                    next_round.append(remaining_teams[i])
                    print(f"  {remaining_teams[i]} gets a bye")
            
            remaining_teams = next_round
            round_num += 1
        
        # Final Play-In match if 2 teams remain
        if len(remaining_teams) == 2:
            print(f"\nPlay-In Final")
            print("-" * 30)
            
            team1, team2 = remaining_teams
            win_prob = self.get_win_probability(team1, team2)
            
            # Simulate BO5 final
            if random.random() < win_prob:
                winner = team1
            else:
                winner = team2
            
            print(f"  {team1} vs {team2}: {winner} advances to Swiss Stage")
            advancing_teams = [winner]
        else:
            advancing_teams = remaining_teams
        
        print(f"\nAdvancing from Play-In: {advancing_teams}")
        return advancing_teams
    
    def run_swiss_stage(self, swiss_teams):
        """
        Run Swiss Stage to determine elimination bracket teams.
        
        Args:
            swiss_teams: List of teams in Swiss Stage
            
        Returns:
            Tuple of (qualified_teams, eliminated_teams, records, seeding)
        """
        print("=" * 60)
        print("SWISS STAGE")
        print("=" * 60)
        print(f"Swiss Stage teams ({len(swiss_teams)}): {swiss_teams}")
        
        # Use proper 5-6-5 seeding distribution for 16 teams
        seed_groups = {}
        for i, team in enumerate(swiss_teams):
            if i < 5:
                seed_groups[team] = 0  # Pool 0 (5 teams)
            elif i < 11:
                seed_groups[team] = 1  # Pool 1 (6 teams)
            else:
                seed_groups[team] = 2  # Pool 2 (5 teams)
        
        # Create probability matrix for Swiss teams
        n_teams = len(swiss_teams)
        win_probs = np.zeros((n_teams, n_teams))
        
        for i, team1 in enumerate(swiss_teams):
            for j, team2 in enumerate(swiss_teams):
                if i == j:
                    win_probs[i, j] = 0.5
                else:
                    win_probs[i, j] = self.get_win_probability(team1, team2)
        
        # Run Swiss Stage
        swiss_tournament = SwissTournament(swiss_teams, seed_groups, self.team_regions, win_probs)
        swiss_results = swiss_tournament.run()
        
        qualified = swiss_results['qualified']
        eliminated = swiss_results['eliminated']
        records = swiss_results['records']
        seeding = swiss_results['seeding']
        
        print(f"\nSwiss Stage Results:")
        print(f"Qualified for Elimination: {qualified}")
        print(f"Eliminated: {eliminated}")
        
        # Print records
        print(f"\nFinal Records:")
        for team, (wins, losses) in records.items():
            status = "QUALIFIED" if team in qualified else "ELIMINATED"
            print(f"  {team:<20} {wins}-{losses} ({status})")
        
        return qualified, eliminated, records, seeding, swiss_results
    
    def run_elimination(self, qualified_teams, seeding):
        """
        Run Elimination bracket to determine champion.
        
        Args:
            qualified_teams: List of 8 teams in elimination
            seeding: Tuple of (top_seeds, mid_seeds, low_seeds)
            
        Returns:
            Tournament results including champion
        """
        print("=" * 60)
        print("ELIMINATION STAGE")
        print("=" * 60)
        print(f"Elimination teams: {qualified_teams}")
        
        if len(qualified_teams) != 8:
            print(f"Warning: Expected 8 teams, got {len(qualified_teams)}")
            # Pad or trim to 8 teams
            if len(qualified_teams) < 8:
                print(f"Warning: Not enough teams qualified. Using {len(qualified_teams)} teams.")
            else:
                qualified_teams = qualified_teams[:8]
        
        # Create probability matrix for elimination teams
        n_teams = len(qualified_teams)
        win_probs = np.zeros((n_teams, n_teams))
        
        for i, team1 in enumerate(qualified_teams):
            for j, team2 in enumerate(qualified_teams):
                if i == j:
                    win_probs[i, j] = 0.5
                else:
                    win_probs[i, j] = self.get_win_probability(team1, team2)
        
        # Create seed groups for elimination (format expected by Elimination class)
        top_seeds, mid_seeds, low_seeds = seeding
        seed_groups = [top_seeds, mid_seeds, low_seeds]
        
        # Run Elimination
        elimination_tournament = Elimination(qualified_teams, seed_groups, win_probs)
        (semi1, semi2), (final1, final2), champion, quarterfinal_results = elimination_tournament.run()
        
        print(f"\nElimination Results:")
        print(f"Quarterfinals:")
        for qf in quarterfinal_results:
            print(f"  {qf['match'][0]} vs {qf['match'][1]} → {qf['winner']}")
        print(f"Semifinals: {semi1} vs {semi2}")
        print(f"Finals: {final1} vs {final2}")
        print(f"🏆 CHAMPION: {champion}")
        
        return {
            'quarterfinals': quarterfinal_results,
            'semifinals': (semi1, semi2),
            'finals': (final1, final2),
            'champion': champion
        }
    
    def simulate_full_tournament(self, verbose=True):
        """
        Simulate complete Worlds tournament with specific format.
        
        Args:
            verbose: Whether to print detailed results
            
        Returns:
            Complete tournament results
        """
        if verbose:
            print("WORLDS 2024 TOURNAMENT SIMULATION")
            print("=" * 60)
        
        # Run Play-In between Invictus Gaming and T1
        playin_teams = ['Invictus Gaming', 'T1']
        playin_win_probs = np.zeros((2, 2))
        for i, team1 in enumerate(playin_teams):
            for j, team2 in enumerate(playin_teams):
                if i == j:
                    playin_win_probs[i, j] = 0.5
                else:
                    playin_win_probs[i, j] = self.get_win_probability(team1, team2)
        
        # Playin already imported at top of file
        playin = Playin(playin_teams, playin_win_probs, best_of=5)
        playin_winner = playin.run()
        playin_loser = playin_teams[1] if playin_winner == playin_teams[0] else playin_teams[0]
        
        if verbose:
            print(f"Play-In Result: {playin_winner} defeats {playin_loser}")
        
        # Determine Swiss teams based on Play-In result
        if playin_winner == 'Invictus Gaming':
            swiss_teams = ['Bilibili Gaming', 'Gen.G eSports', 'G2 Esports', 'FlyQuest', 
                          'CTBC Flying Oyster', 'Anyone s Legend', 'Hanwha Life eSports', 
                          'Movistar KOI', 'Vivo Keyd Stars', 'Team Secret Whales', 
                          'Top Esports', 'Invictus Gaming', 'KT Rolster', 'Fnatic', 
                          '100 Thieves', 'PSG Talon']
        else:  # T1 wins
            swiss_teams = ['Bilibili Gaming', 'Gen.G eSports', 'G2 Esports', 'FlyQuest', 
                          'CTBC Flying Oyster', 'Anyone s Legend', 'Hanwha Life eSports', 
                          'Movistar KOI', 'Vivo Keyd Stars', 'Team Secret Whales', 
                          'KT Rolster', 'T1', 'Top Esports', 'Fnatic', 
                          '100 Thieves', 'PSG Talon']
        
        # Run Swiss Stage
        qualified, eliminated, records, seeding, swiss_details = self.run_swiss_stage(swiss_teams)
        
        # Run Elimination
        elimination_results = self.run_elimination(qualified, seeding)
        
        return {
            'playin_winner': playin_winner,
            'playin_loser': playin_loser,
            'swiss_teams': swiss_teams,
            'swiss_qualified': qualified,
            'swiss_eliminated': eliminated,
            'swiss_records': records,
            'swiss_details': swiss_details,
            'elimination_results': elimination_results,
            'champion': elimination_results['champion']
        }
    
    def run_multiple_simulations(self, num_simulations=100, num_teams=16):
        """
        Run multiple tournament simulations for statistics.
        
        Args:
            num_simulations: Number of simulations to run
            num_teams: Number of teams per tournament
            
        Returns:
            Simulation statistics
        """
        print(f"Running {num_simulations} tournament simulations...")
        
        champions = []
        all_results = []
        
        for i in range(num_simulations):
            if (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{num_simulations} simulations")
            
            # Run simulation without verbose output
            result = self.simulate_full_tournament(verbose=False)
            champions.append(result['champion'])
            all_results.append(result)
        
        # Calculate championship statistics
        champion_counts = {}
        for champion in champions:
            champion_counts[champion] = champion_counts.get(champion, 0) + 1
        
        # Sort by win frequency
        champion_stats = [(team, count, count/num_simulations) 
                         for team, count in champion_counts.items()]
        champion_stats.sort(key=lambda x: x[1], reverse=True)
        
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        print(f"Total simulations: {num_simulations}")
        print("\nChampionship probabilities:")
        
        for i, (team, wins, probability) in enumerate(champion_stats):
            print(f"  {i+1:2d}. {team:<25} {wins:3d} wins ({probability:.1%})")
        
        return {
            'num_simulations': num_simulations,
            'champion_stats': champion_stats,
            'all_results': all_results
        }

def main():
    """Main function to run tournament simulation"""
    # Set random seed for reproducible results (optional)
    random.seed(42)
    np.random.seed(42)
    
    # Initialize tournament
    worlds = WorldsTournament()
    
    print("Worlds Tournament Simulation")
    print("Choose simulation mode:")
    print("1. Single tournament (16 teams)")
    print("2. Multiple simulations (100x)")
    print("3. Quick demo (8 teams)")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
    except:
        choice = "1"  # Default
    
    if choice == "1":
        # Single full tournament
        result = worlds.simulate_full_tournament()
        
    elif choice == "2":
        # Multiple simulations
        stats = worlds.run_multiple_simulations(100)
        
        # Save results
        import pandas as pd
        results_df = pd.DataFrame(stats['champion_stats'], 
                                columns=['Team', 'Championships', 'Win_Rate'])
        results_df['Region'] = results_df['Team'].map(worlds.team_regions)
        results_df.to_csv('dataset/worlds_simulation_results.csv', index=False)
        print(f"\nResults saved to dataset/worlds_simulation_results.csv")
        
    else:
        # Quick demo - single tournament
        print("Running single tournament...")
        result = worlds.simulate_full_tournament()

if __name__ == "__main__":
    main()