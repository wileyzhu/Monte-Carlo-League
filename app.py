from flask import Flask, render_template, request, jsonify, session
import json
import random
import numpy as np
from src.tournament.worlds_tournament import WorldsTournament
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'worlds2025_simulation_key'

# Global tournament instance
tournament = None

def initialize_tournament():
    """Initialize the tournament instance"""
    global tournament
    if tournament is None:
        tournament = WorldsTournament()
    return tournament



@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/playin')
def playin():
    """Play-in simulation page"""
    return render_template('playin.html')

@app.route('/swiss')
def swiss():
    """Swiss stage simulation page"""
    return render_template('swiss.html')

@app.route('/api/simulate', methods=['POST'])
def simulate_tournament():
    """Run tournament simulation"""
    try:
        data = request.get_json()
        simulation_type = data.get('type', 'single')
        num_simulations = data.get('num_simulations', 1)
        
        # Initialize tournament
        worlds = initialize_tournament()
        
        if simulation_type == 'single':
            # Single tournament simulation
            result = worlds.simulate_full_tournament(verbose=False)
            
            return jsonify({
                'success': True,
                'type': 'single',
                'result': {
                    'champion': result['champion'],
                    'champion_region': worlds.team_regions.get(result['champion'], 'Unknown'),
                    'playin_winner': result['playin_winner'],
                    'playin_loser': result['playin_loser'],
                    'swiss_qualified': result['swiss_qualified'],
                    'swiss_eliminated': result['swiss_eliminated'],
                    'swiss_records': result['swiss_records'],

                    'elimination_results': result['elimination_results']
                }
            })
            
        elif simulation_type == 'multiple':
            # Multiple simulations
            num_sims = min(max(int(num_simulations), 1), 1000)  # Cap at 1000
            
            champions = []
            all_results = []
            
            for i in range(num_sims):
                result = worlds.simulate_full_tournament(verbose=False)
                champions.append(result['champion'])
                all_results.append(result)
            
            # Calculate statistics
            champion_counts = {}
            for champion in champions:
                champion_counts[champion] = champion_counts.get(champion, 0) + 1
            
            # Sort by win frequency
            champion_stats = [(team, count, count/num_sims) 
                             for team, count in champion_counts.items()]
            champion_stats.sort(key=lambda x: x[1], reverse=True)
            
            # Regional statistics
            region_wins = {}
            for champion in champions:
                region = worlds.team_regions.get(champion, 'Unknown')
                region_wins[region] = region_wins.get(region, 0) + 1
            
            return jsonify({
                'success': True,
                'type': 'multiple',
                'num_simulations': num_sims,
                'champion_stats': champion_stats,
                'regional_distribution': region_wins,
                'most_likely_champion': champion_stats[0][0] if champion_stats else None,
                'champion_probability': champion_stats[0][2] if champion_stats else 0
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/playin/predictions')
def playin_predictions():
    """Get play-in predictions and team analysis"""
    try:
        worlds = initialize_tournament()
        
        # Calculate REAL win probability using the probability matrix
        win_prob_t1 = worlds.get_win_probability('T1', 'Invictus Gaming')
        win_prob_ig = 1 - win_prob_t1
        
        # Get team performance from probability matrix (average win rate against all teams)
        t1_performance = worlds.prob_matrix_df.loc['T1'].mean()
        ig_performance = worlds.prob_matrix_df.loc['Invictus Gaming'].mean()
        
        # Get REAL team rankings based on average performance
        team_rankings = []
        for team in worlds.all_teams:
            avg_performance = worlds.prob_matrix_df.loc[team].mean()
            team_rankings.append((team, avg_performance))
        team_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Find actual ranks
        t1_rank = next((i+1 for i, (team, _) in enumerate(team_rankings) if team == 'T1'), 'Unknown')
        ig_rank = next((i+1 for i, (team, _) in enumerate(team_rankings) if team == 'Invictus Gaming'), 'Unknown')
        
        team_analysis = {
            'T1': {
                'region': worlds.team_regions.get('T1', 'LCK'),
                'rank': t1_rank,
                'predicted_performance': round(t1_performance, 3),
                'expected_win_rate': round(win_prob_t1, 3),
                'description': f'Performance rating: {t1_performance:.3f} - Ranked #{t1_rank} globally'
            },
            'Invictus Gaming': {
                'region': worlds.team_regions.get('Invictus Gaming', 'LPL'),
                'rank': ig_rank,
                'predicted_performance': round(ig_performance, 3),
                'expected_win_rate': round(win_prob_ig, 3),
                'description': f'Performance rating: {ig_performance:.3f} - Ranked #{ig_rank} globally'
            }
        }
        
        return jsonify({
            'success': True,
            'matchup': {
                'head_to_head_probability': {
                    'T1': f'{win_prob_t1*100:.1f}%',
                    'Invictus Gaming': f'{win_prob_ig*100:.1f}%'
                }
            },
            'team_analysis': team_analysis,
            'series_format': 'Best of 5',
            'prediction_confidence': f'Model-based ({max(win_prob_t1, win_prob_ig)*100:.1f}%)'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/playin/simulate', methods=['POST'])
def simulate_playin():
    """Run play-in simulation with realistic BO5 score distributions"""
    try:
        data = request.get_json()
        simulation_type = data.get('type', 'single')
        num_simulations = data.get('num_simulations', 1)
        
        worlds = initialize_tournament()
        
        if simulation_type == 'single':
            # Single play-in simulation using realistic playin module
            play_in_teams = ['T1', 'Invictus Gaming']
            
            # Create win probability matrix for playin teams
            playin_win_probs = np.zeros((2, 2))
            for i, team1 in enumerate(play_in_teams):
                for j, team2 in enumerate(play_in_teams):
                    if i == j:
                        playin_win_probs[i, j] = 0.5
                    else:
                        playin_win_probs[i, j] = worlds.get_win_probability(team1, team2)
            
            # Run realistic playin simulation
            from src.tournament.playin import Playin
            playin = Playin(play_in_teams, playin_win_probs, best_of=5)
            winner = playin.run(realistic=True)
            series_details = playin.get_series_details()
            
            # Get real win probabilities for display
            t1_win_prob = worlds.get_win_probability('T1', 'Invictus Gaming')
            ig_win_prob = 1 - t1_win_prob
            
            return jsonify({
                'success': True,
                'result': {
                    'winner': series_details['winner'],
                    'loser': series_details['loser'],
                    'final_score': series_details['final_score'],
                    'series_length': series_details['series_length'],
                    'games': series_details['games'],
                    'upset': series_details['winner'] == 'Invictus Gaming',
                    'win_probability_used': series_details['win_probability'],
                    'teams': {
                        'team_a': {
                            'name': 'T1',
                            'win_percentage': f'{t1_win_prob*100:.1f}%'
                        },
                        'team_b': {
                            'name': 'Invictus Gaming', 
                            'win_percentage': f'{ig_win_prob*100:.1f}%'
                        }
                    }
                }
            })
        else:
            # Multiple play-in simulations with realistic score distributions
            winners = []
            all_results = []
            
            # Get win probabilities
            t1_win_prob = worlds.get_win_probability('T1', 'Invictus Gaming')
            
            for _ in range(num_simulations):
                play_in_teams = ['T1', 'Invictus Gaming']
                
                # Create win probability matrix
                playin_win_probs = np.zeros((2, 2))
                for i, team1 in enumerate(play_in_teams):
                    for j, team2 in enumerate(play_in_teams):
                        if i == j:
                            playin_win_probs[i, j] = 0.5
                        else:
                            playin_win_probs[i, j] = worlds.get_win_probability(team1, team2)
                
                # Run realistic playin simulation
                from src.tournament.playin import Playin
                playin = Playin(play_in_teams, playin_win_probs, best_of=5)
                winner = playin.run(realistic=True)
                series_details = playin.get_series_details()
                
                winners.append(winner)
                all_results.append({
                    'winner': series_details['winner'],
                    'final_score': series_details['final_score'],
                    'upset': series_details['winner'] == 'Invictus Gaming',
                    'series_length': series_details['series_length']
                })
            
            # Calculate statistics
            t1_wins = winners.count('T1')
            ig_wins = winners.count('Invictus Gaming')
            
            # Calculate comprehensive score distribution
            score_distribution = {}
            score_by_winner = {'T1': {}, 'Invictus Gaming': {}}
            
            for result in all_results:
                score = result['final_score']
                winner = result['winner']
                
                # Overall distribution
                score_distribution[score] = score_distribution.get(score, 0) + 1
                
                # Distribution by winner
                if score not in score_by_winner[winner]:
                    score_by_winner[winner][score] = 0
                score_by_winner[winner][score] += 1
            
            # Convert to percentages
            score_dist_percent = {}
            for score, count in score_distribution.items():
                score_dist_percent[score] = (count / num_simulations) * 100
            
            # Convert winner distributions to percentages
            score_by_winner_percent = {}
            for winner in ['T1', 'Invictus Gaming']:
                score_by_winner_percent[winner] = {}
                for score, count in score_by_winner[winner].items():
                    score_by_winner_percent[winner][score] = (count / num_simulations) * 100
            
            return jsonify({
                'success': True,
                'result': {
                    'num_simulations': num_simulations,
                    'most_likely_winner': 'T1' if t1_wins > ig_wins else 'Invictus Gaming',
                    'confidence': max(t1_wins, ig_wins) / num_simulations,
                    'score_distribution': score_dist_percent,
                    'score_by_winner': score_by_winner_percent,
                    'teams': {
                        'team_a': {
                            'name': 'T1',
                            'wins': t1_wins,
                            'win_percentage': f"{(t1_wins/num_simulations*100):.1f}%",
                            'predicted_win_rate': t1_win_prob,
                            'actual_win_rate': t1_wins/num_simulations
                        },
                        'team_b': {
                            'name': 'Invictus Gaming',
                            'wins': ig_wins, 
                            'win_percentage': f"{(ig_wins/num_simulations*100):.1f}%",
                            'predicted_win_rate': 1 - t1_win_prob,
                            'actual_win_rate': ig_wins/num_simulations
                        }
                    },
                    'all_results': all_results[:10]  # Show last 10 results
                }
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/simulate_swiss', methods=['POST'])
def simulate_swiss():
    """Run Swiss stage simulation"""
    try:
        data = request.get_json()
        simulation_type = data.get('type', 'single')
        num_simulations = data.get('num_simulations', 1)
        
        worlds = initialize_tournament()
        
        if simulation_type == 'single':
            # Single Swiss simulation
            swiss_result = worlds.simulate_swiss_stage(verbose=False)
            
            return jsonify({
                'success': True,
                'type': 'single',
                'result': {
                    'qualified': swiss_result['qualified'],
                    'eliminated': swiss_result['eliminated'],
                    'records': swiss_result.get('records', {}),
                    'matches': swiss_result.get('matches', [])
                }
            })
        else:
            # Multiple Swiss simulations
            all_qualified = []
            for _ in range(num_simulations):
                result = worlds.simulate_swiss_stage(verbose=False)
                all_qualified.extend(result['qualified'])
            
            # Calculate qualification statistics
            qual_counts = {}
            for team in all_qualified:
                qual_counts[team] = qual_counts.get(team, 0) + 1
            
            qual_stats = [(team, count, count/num_simulations) 
                         for team, count in qual_counts.items()]
            qual_stats.sort(key=lambda x: x[1], reverse=True)
            
            return jsonify({
                'success': True,
                'type': 'multiple',
                'num_simulations': num_simulations,
                'qualification_stats': qual_stats
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/swiss/teams')
def get_swiss_teams():
    """Get Swiss stage team pool"""
    try:
        playin_winner = request.args.get('playin_winner', 'T1')
        worlds = initialize_tournament()
        
        # Get all teams for Swiss stage (15 + playin winner)
        swiss_teams = [
            'Bilibili Gaming', 'Gen.G eSports', 'G2 Esports', 'FlyQuest',
            'CTBC Flying Oyster', 'Anyone s Legend', 'Hanwha Life eSports',
            'Movistar KOI', 'Vivo Keyd Stars', 'Team Secret Whales',
            'KT Rolster', 'Top Esports', 'Fnatic', '100 Thieves', 'PSG Talon'
        ]
        
        # Add playin winner
        swiss_teams.append(playin_winner)
        
        # Create seed groups (5-6-5 distribution)
        seed_groups = {}
        for i, team in enumerate(swiss_teams):
            if i < 5:
                seed_groups[team] = 0  # Pool 0 (5 teams)
            elif i < 11:
                seed_groups[team] = 1  # Pool 1 (6 teams)
            else:
                seed_groups[team] = 2  # Pool 2 (5 teams)
        
        # Group by region
        teams_by_region = {}
        for team in swiss_teams:
            region = worlds.team_regions.get(team, 'Unknown')
            if region not in teams_by_region:
                teams_by_region[region] = []
            teams_by_region[region].append(team)
        
        # Format teams with region info for frontend
        teams_formatted = []
        for team in swiss_teams:
            teams_formatted.append({
                'name': team,
                'region': worlds.team_regions.get(team, 'Unknown')
            })
        
        return jsonify({
            'success': True,
            'teams': teams_formatted,
            'teams_by_region': teams_by_region,
            'seed_groups': seed_groups,
            'playin_winner': playin_winner,
            'total_teams': len(swiss_teams)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/swiss/simulate', methods=['POST'])
def simulate_swiss_stage():
    """Run Swiss stage simulation"""
    try:
        data = request.get_json()
        simulation_type = data.get('type', 'single')
        playin_winner = data.get('playin_winner', 'T1')
        seeding_strategy = data.get('seeding_strategy', 'power_rankings')
        num_simulations = data.get('num_simulations', 1)
        
        worlds = initialize_tournament()
        
        if simulation_type == 'single':
            # Single Swiss simulation with detailed tracking
            swiss_teams = [
                'Bilibili Gaming', 'Gen.G eSports', 'G2 Esports', 'FlyQuest',
                'CTBC Flying Oyster', 'Anyone s Legend', 'Hanwha Life eSports',
                'Movistar KOI', 'Vivo Keyd Stars', 'Team Secret Whales',
                'KT Rolster', 'Top Esports', 'Fnatic', '100 Thieves', 'PSG Talon',
                playin_winner
            ]
            
            qualified_teams, eliminated_teams, records, seeding, swiss_details = worlds.run_swiss_stage(swiss_teams)
            
            # Convert records to serializable format
            records_formatted = {}
            for team, (wins, losses) in records.items():
                records_formatted[team] = [wins, losses]
            
            return jsonify({
                'success': True,
                'type': 'single',
                'result': {
                    'qualified': qualified_teams,
                    'eliminated': eliminated_teams,
                    'records': records_formatted,
                    'round_results': swiss_details['round_results'],
                    'total_rounds': swiss_details['total_rounds'],
                    'playin_winner': playin_winner,
                    'seeding_strategy': seeding_strategy
                }
            })
        else:
            # Multiple simulations with different analysis types
            analysis_type = data.get('analysis_type', 'qualification')
            
            if analysis_type == 'championship':
                # Use proper WorldsTournament for championship analysis FROM ROUND 2 STATE
                # State BEFORE Round 3 (after Round 2 completed) - predict Round 3 outcomes
                round2_records = {
                    # 2-0 teams (after Round 2)
                    'KT Rolster': [2, 0],           # Round 1: W, Round 2: W (beat TSW)
                    'Top Esports': [2, 0],          # Round 1: W, Round 2: W (beat 100)
                    'CTBC Flying Oyster': [2, 0],   # Round 1: W, Round 2: W (beat T1)
                    'Anyone s Legend': [2, 0],      # Round 1: W, Round 2: W (beat GEN)
                    
                    # 1-1 teams (after Round 2)
                    'Team Secret Whales': [1, 1],   # Round 1: W, Round 2: L (lost to KT)
                    'FlyQuest': [1, 1],             # Round 1: L, Round 2: W (beat VKS)
                    'Gen.G eSports': [1, 1],        # Round 1: W, Round 2: L (lost to AL)
                    'T1': [1, 1],                   # Round 1: W, Round 2: L (lost to CFO)
                    'G2 Esports': [1, 1],           # Round 1: L, Round 2: W (beat KOI)
                    'Bilibili Gaming': [1, 1],      # Round 1: L, Round 2: W (beat FNC)
                    '100 Thieves': [1, 1],          # Round 1: W, Round 2: L (lost to TES)
                    'Hanwha Life eSports': [1, 1],  # Round 1: L, Round 2: W (beat PSG)
                    
                    # 0-2 teams (after Round 2)
                    'Movistar KOI': [0, 2],         # Round 1: L, Round 2: L (lost to G2)
                    'Fnatic': [0, 2],               # Round 1: L, Round 2: L (lost to BLG)
                    'Vivo Keyd Stars': [0, 2],      # Round 1: L, Round 2: L (lost to FLY)
                    'PSG Talon': [0, 2]             # Round 1: L, Round 2: L (lost to HLE)
                }
                
                # Create the real results state for simulate_from_real_results
                real_results_state = {
                    'playin_completed': True,
                    'playin_winner': playin_winner,
                    'swiss_completed': False,
                    'swiss_current_records': round2_records,
                    'swiss_round': 3,  # Starting from Round 3 (Round 2 completed)
                    'elimination_completed': False
                }
                
                # Use the proper method to simulate from Round 1 state
                championship_results = worlds.simulate_from_real_results(real_results_state, num_simulations)
                
                # Extract championship distribution
                record_distribution = {}
                for team, count, probability in championship_results['champion_stats']:
                    record_distribution[team] = count
                
                result = {
                    'num_simulations': num_simulations,
                    'record_distribution': record_distribution,
                    'playin_winner': playin_winner,
                    'seeding_strategy': seeding_strategy
                }
                
                return jsonify({
                    'success': True,
                    'type': 'multiple',
                    'result': result
                })
            
            else:
                # For other analysis types, use the existing Swiss simulation logic
                # State BEFORE Round 3 (after Round 2 completed) - predict Round 3 outcomes
                round2_records = {
                    # 2-0 teams (after Round 2)
                    'KT Rolster': [2, 0],           # Round 1: W, Round 2: W (beat TSW)
                    'Top Esports': [2, 0],          # Round 1: W, Round 2: W (beat 100)
                    'CTBC Flying Oyster': [2, 0],   # Round 1: W, Round 2: W (beat T1)
                    'Anyone s Legend': [2, 0],      # Round 1: W, Round 2: W (beat GEN)
                    
                    # 1-1 teams (after Round 2)
                    'Team Secret Whales': [1, 1],   # Round 1: W, Round 2: L (lost to KT)
                    'FlyQuest': [1, 1],             # Round 1: L, Round 2: W (beat VKS)
                    'Gen.G eSports': [1, 1],        # Round 1: W, Round 2: L (lost to AL)
                    'T1': [1, 1],                   # Round 1: W, Round 2: L (lost to CFO)
                    'G2 Esports': [1, 1],           # Round 1: L, Round 2: W (beat KOI)
                    'Bilibili Gaming': [1, 1],      # Round 1: L, Round 2: W (beat FNC)
                    '100 Thieves': [1, 1],          # Round 1: W, Round 2: L (lost to TES)
                    'Hanwha Life eSports': [1, 1],  # Round 1: L, Round 2: W (beat PSG)
                    
                    # 0-2 teams (after Round 2)
                    'Movistar KOI': [0, 2],         # Round 1: L, Round 2: L (lost to G2)
                    'Fnatic': [0, 2],               # Round 1: L, Round 2: L (lost to BLG)
                    'Vivo Keyd Stars': [0, 2],      # Round 1: L, Round 2: L (lost to FLY)
                    'PSG Talon': [0, 2]             # Round 1: L, Round 2: L (lost to HLE)
                }
                
                swiss_teams = list(round2_records.keys())
                
                all_qualified = []
                qualification_counts = {}
                regional_stats = {}
                
                for _ in range(num_simulations):
                    # Start with Round 2 records
                    current_records = {team: record.copy() for team, record in round2_records.items()}
                    
                    # ROUND 3: Use ACTUAL matchup draws (not random pairing)
                    round3_matches = [
                        # High (2-0 vs 2-0) - BO3
                        ('KT Rolster', 'Top Esports'),
                        ('CTBC Flying Oyster', 'Anyone s Legend'),
                        # Middle (1-1 vs 1-1) - BO1
                        ('Team Secret Whales', 'FlyQuest'),
                        ('Gen.G eSports', 'T1'),
                        ('G2 Esports', 'Bilibili Gaming'),
                        ('100 Thieves', 'Hanwha Life eSports'),
                        # Low (0-2 vs 0-2) - BO3
                        ('Movistar KOI', 'Fnatic'),
                        ('Vivo Keyd Stars', 'PSG Talon')
                    ]
                    
                    # Simulate Round 3 with actual draws
                    for team1, team2 in round3_matches:
                        game_win_prob = worlds.get_win_probability(team1, team2)
                        
                        # Check if this is a BO3 match (2-0 vs 2-0 or 0-2 vs 0-2)
                        wins1, losses1 = current_records[team1]
                        wins2, losses2 = current_records[team2]
                        is_bo3 = (wins1 == 2 and losses1 == 0 and wins2 == 2 and losses2 == 0) or \
                                 (wins1 == 0 and losses1 == 2 and wins2 == 0 and losses2 == 2)
                        
                        if is_bo3:
                            # Calculate BO3 win probability
                            bo3_win_prob = (game_win_prob ** 2) + (2 * (game_win_prob ** 2) * (1 - game_win_prob))
                            if random.random() < bo3_win_prob:
                                winner, loser = team1, team2
                            else:
                                winner, loser = team2, team1
                        else:
                            # BO1
                            if random.random() < game_win_prob:
                                winner, loser = team1, team2
                            else:
                                winner, loser = team2, team1
                        
                        # Update records
                        current_records[winner][0] += 1  # Add win
                        current_records[loser][1] += 1   # Add loss
                    
                    # Continue Swiss stage from Round 4 onwards (if needed)
                    round_num = 4
                    
                    while not all(wins >= 3 or losses >= 3 for wins, losses in current_records.values()):
                        # Determine matchups for remaining rounds (Swiss pairing)
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
                            win_prob = worlds.get_win_probability(team1, team2)
                            
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
                    qualified_teams = [team for team, (wins, losses) in current_records.items() if wins >= 3]
                    all_qualified.append(qualified_teams)
                    
                    # Count qualifications
                    for team in qualified_teams:
                        qualification_counts[team] = qualification_counts.get(team, 0) + 1
                    
                    # Regional analysis
                    if analysis_type == 'regional':
                        for team in qualified_teams:
                            region = worlds.team_regions.get(team, 'Unknown')
                            if region not in regional_stats:
                                regional_stats[region] = {'qualified_count': 0, 'total_teams': 0}
                            regional_stats[region]['qualified_count'] += 1
                        
                        # Count total teams per region
                        for team in swiss_teams:
                            region = worlds.team_regions.get(team, 'Unknown')
                            if region not in regional_stats:
                                regional_stats[region] = {'qualified_count': 0, 'total_teams': 0}
                            if regional_stats[region]['total_teams'] == 0:
                                # Count teams in this region
                                regional_stats[region]['total_teams'] = sum(1 for t in swiss_teams if worlds.team_regions.get(t) == region)
            
            # Calculate qualification statistics
            qual_stats = []
            for team, count in qualification_counts.items():
                region = worlds.team_regions.get(team, 'Unknown')
                qual_stats.append({
                    'team': team,
                    'region': region,
                    'qualifications': count,
                    'percentage': (count / num_simulations) * 100
                })
            
            qual_stats.sort(key=lambda x: x['percentage'], reverse=True)
            
            # Format regional stats
            formatted_regional_stats = {}
            for region, stats in regional_stats.items():
                formatted_regional_stats[region] = {
                    'team_count': stats['total_teams'],
                    'avg_qualified': stats['qualified_count'] / num_simulations,
                    'qualification_rate': (stats['qualified_count'] / (stats['total_teams'] * num_simulations)) * 100
                }
            
            result = {
                'num_simulations': num_simulations,
                'qualification_stats': qual_stats,
                'playin_winner': playin_winner,
                'seeding_strategy': seeding_strategy
            }
            
            if analysis_type == 'regional':
                result['regional_stats'] = formatted_regional_stats
            
            return jsonify({
                'success': True,
                'type': 'multiple',
                'result': result
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/teams')
def get_teams():
    """Get all teams and their regions"""
    try:
        worlds = initialize_tournament()
        
        teams_by_region = {}
        for team, region in worlds.team_regions.items():
            if region not in teams_by_region:
                teams_by_region[region] = []
            teams_by_region[region].append(team)
        
        return jsonify({
            'success': True,
            'teams_by_region': teams_by_region,
            'all_teams': list(worlds.team_regions.keys())
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/team_rankings')
def get_team_rankings():
    """Get individual team power rankings"""
    try:
        worlds = initialize_tournament()
        
        # Get all teams and calculate their average win probability
        team_rankings = []
        all_teams = list(worlds.team_regions.keys())
        
        for team in all_teams:
            total_win_prob = 0
            opponent_count = 0
            
            for opponent in all_teams:
                if team != opponent:
                    win_prob = worlds.get_win_probability(team, opponent)
                    total_win_prob += win_prob
                    opponent_count += 1
            
            avg_win_prob = total_win_prob / opponent_count if opponent_count > 0 else 0.5
            region = worlds.team_regions.get(team, 'Unknown')
            
            team_rankings.append({
                'team': team,
                'region': region,
                'power_rating': avg_win_prob,
                'percentage': avg_win_prob * 100
            })
        
        # Sort by power rating (highest first)
        team_rankings.sort(key=lambda x: x['power_rating'], reverse=True)
        
        # Add ranking numbers
        for i, team_data in enumerate(team_rankings):
            team_data['rank'] = i + 1
        
        return jsonify({
            'success': True,
            'team_rankings': team_rankings
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/regional_strengths')
def get_regional_strengths():
    """Get MSI-based regional strength data"""
    try:
        # Regional strengths from MSI 2025 + EWC 2025 combined (calculated from actual results)
        regional_strengths = {
            'LCK': {'strength': 1.000, 'description': 'Completely dominant performance (11-3 matches, 65.9% games, 80.0% weighted)'},
            'LPL': {'strength': 0.821, 'description': 'Strong but clearly behind Korea (8-5 matches, 59.2% games, 62.2% weighted)'},
            'LEC': {'strength': 0.601, 'description': 'Competitive middle tier (8-8 matches, 38.6% games, 48.3% weighted)'},
            'LCP': {'strength': 0.393, 'description': 'Struggled significantly (2-7 matches, 37.9% games, 23.5% weighted)'},
            'LTA': {'strength': 0.314, 'description': 'Weakest region overall (1-7 matches, 38.5% games, 13.3% weighted)'}
        }
        
        return jsonify({
            'success': True,
            'regional_strengths': regional_strengths
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@app.route('/api/simulate_from_real', methods=['POST'])
def simulate_from_real_results():
    """Simulate tournament from real results input"""
    try:
        data = request.get_json()
        real_results = data.get('real_results', {})
        num_simulations = data.get('num_simulations', 100)
        
        worlds = initialize_tournament()
        
        # Run simulation from real results
        results = worlds.simulate_from_real_results(real_results, num_simulations)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/swiss/simulate_from_draw', methods=['POST'])
def simulate_swiss_from_first_draw():
    """Simulate Swiss stage from first draw matchups"""
    try:
        data = request.get_json()
        first_draw_matchups = data.get('first_draw_matchups', [])
        playin_winner = data.get('playin_winner', 'T1')
        num_simulations = data.get('num_simulations', 100)
        simulation_type = data.get('simulation_type', 'qualification')
        
        worlds = initialize_tournament()
        
        if simulation_type == 'matchup_results':
            # Simulate only Round 1 matchup results
            results = worlds.simulate_matchup_results(first_draw_matchups, playin_winner, num_simulations)
        else:
            # Simulate full Swiss stage for qualification rates
            results = worlds.simulate_swiss_from_first_draw(first_draw_matchups, playin_winner, num_simulations)
        
        return jsonify({
            'success': True,
            'result': results,
            'simulation_type': simulation_type
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/swiss/simulate_from_round1', methods=['POST'])
def simulate_from_round1_results():
    """Simulate Swiss stage from actual Round 1 results using proper simulation"""
    try:
        data = request.get_json()
        num_simulations = data.get('num_simulations', 200)
        
        worlds = initialize_tournament()
        
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
        
        swiss_teams = list(round1_records.keys())
        
        # Run proper simulations from Round 1 state
        qualification_counts = {}
        
        for sim in range(num_simulations):
            # Start with Round 1 records
            current_records = {team: record.copy() for team, record in round1_records.items()}
            
            # Continue Swiss stage from Round 2
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
                    win_prob = worlds.get_win_probability(team1, team2)
                    
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
        
        # Calculate results
        qual_stats = []
        for team, count in qualification_counts.items():
            percentage = (count / num_simulations) * 100
            region = worlds.team_regions.get(team, 'Unknown')
            qual_stats.append({
                'team': team,
                'region': region,
                'qualifications': count,
                'percentage': percentage
            })
        
        qual_stats.sort(key=lambda x: x['percentage'], reverse=True)
        
        # Separate into categories
        likely_qualified = [stat for stat in qual_stats if stat['percentage'] > 50]
        bubble_teams = [stat for stat in qual_stats if 20 <= stat['percentage'] <= 50]
        
        # Calculate regional analysis
        regional_quals = {}
        for stat in qual_stats:
            region = stat['region']
            if region not in regional_quals:
                regional_quals[region] = {'total_percentage': 0, 'teams': []}
            regional_quals[region]['total_percentage'] += stat['percentage']
            regional_quals[region]['teams'].append(f"{stat['team']} ({stat['percentage']:.1f}%)")
        
        results = {
            'qualification_stats': qual_stats,
            'regional_analysis': regional_quals,
            'likely_qualified': likely_qualified,
            'bubble_teams': bubble_teams,
            'num_simulations': num_simulations
        }
        
        return jsonify({
            'success': True,
            'result': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/swiss/simulate_round2_matchups', methods=['POST'])
def simulate_round2_matchups():
    """Simulate Round 2 matchup outcomes"""
    try:
        data = request.get_json()
        num_simulations = data.get('num_simulations', 100)
        round2_matchups = data.get('round2_matchups', [])
        
        worlds = initialize_tournament()
        
        matchup_results = {}
        
        for matchup in round2_matchups:
            team_a = matchup['team_a']
            team_b = matchup['team_b']
            
            # Get win probability from matrix
            win_prob_a = worlds.get_win_probability(team_a, team_b)
            
            # Simulate the matchup
            team_a_wins = 0
            for _ in range(num_simulations):
                if random.random() < win_prob_a:
                    team_a_wins += 1
            
            team_b_wins = num_simulations - team_a_wins
            
            matchup_key = f"{team_a} vs {team_b}"
            matchup_results[matchup_key] = {
                'team_a': team_a,
                'team_b': team_b,
                'team_a_wins': team_a_wins,
                'team_b_wins': team_b_wins,
                'team_a_percentage': (team_a_wins / num_simulations) * 100,
                'team_b_percentage': (team_b_wins / num_simulations) * 100,
                'expected_prob_a': win_prob_a * 100,
                'expected_prob_b': (1 - win_prob_a) * 100
            }
        
        return jsonify({
            'success': True,
            'result': {
                'matchup_results': matchup_results,
                'num_simulations': num_simulations
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/swiss/match_predictions', methods=['POST'])
def get_match_predictions():
    """Get Round 3 match predictions with real probabilities"""
    try:
        worlds = initialize_tournament()
        
        # Round 3 matchups based on actual draw (after Round 2 completed)
        # High matches (2-0 vs 2-0) - BO3
        high_matches = [
            {'team_a': 'KT Rolster', 'team_b': 'Top Esports', 'format': 'BO3'},
            {'team_a': 'CTBC Flying Oyster', 'team_b': 'Anyone s Legend', 'format': 'BO3'}
        ]
        
        # Middle matches (1-1 vs 1-1) - BO1
        middle_matches = [
            {'team_a': 'Team Secret Whales', 'team_b': 'FlyQuest', 'format': 'BO1'},
            {'team_a': 'Gen.G eSports', 'team_b': 'T1', 'format': 'BO1'},
            {'team_a': 'G2 Esports', 'team_b': 'Bilibili Gaming', 'format': 'BO1'},
            {'team_a': '100 Thieves', 'team_b': 'Hanwha Life eSports', 'format': 'BO1'}
        ]
        
        # Low matches (0-2 vs 0-2) - BO3
        low_matches = [
            {'team_a': 'Movistar KOI', 'team_b': 'Fnatic', 'format': 'BO3'},
            {'team_a': 'Vivo Keyd Stars', 'team_b': 'PSG Talon', 'format': 'BO3'}
        ]
        
        # Calculate real probabilities for high matches (BO3)
        for match in high_matches:
            game_win_prob = worlds.get_win_probability(match['team_a'], match['team_b'])
            # BO3 win probability
            bo3_win_prob = calculate_bo3_win_probability(game_win_prob)
            match['team_a_prob'] = bo3_win_prob * 100
            match['team_b_prob'] = (1 - bo3_win_prob) * 100
        
        # Calculate real probabilities for middle matches (BO1)
        for match in middle_matches:
            win_prob_a = worlds.get_win_probability(match['team_a'], match['team_b'])
            match['team_a_prob'] = win_prob_a * 100
            match['team_b_prob'] = (1 - win_prob_a) * 100
        
        # Calculate real probabilities for low matches (BO3)
        for match in low_matches:
            game_win_prob = worlds.get_win_probability(match['team_a'], match['team_b'])
            # BO3 win probability
            bo3_win_prob = calculate_bo3_win_probability(game_win_prob)
            match['team_a_prob'] = bo3_win_prob * 100
            match['team_b_prob'] = (1 - bo3_win_prob) * 100
        
        # Find most competitive and biggest favorite
        all_matches = high_matches + middle_matches + low_matches
        
        most_competitive = min(all_matches, key=lambda m: abs(m['team_a_prob'] - 50))
        most_competitive_diff = abs(most_competitive['team_a_prob'] - 50) * 2
        
        biggest_favorite = max(all_matches, key=lambda m: abs(m['team_a_prob'] - 50))
        biggest_favorite_diff = abs(biggest_favorite['team_a_prob'] - 50) * 2
        
        result = {
            'high_matches': high_matches,
            'middle_matches': middle_matches,
            'low_matches': low_matches,
            'most_competitive': {
                'matchup': f"{most_competitive['team_a']} vs {most_competitive['team_b']}",
                'difference': most_competitive_diff
            },
            'biggest_favorite': {
                'matchup': f"{biggest_favorite['team_a']} vs {biggest_favorite['team_b']}",
                'difference': biggest_favorite_diff
            }
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Removed simulate_elimination_bracket - now using proper WorldsTournament.run_multiple_simulations()

def calculate_bo3_win_probability(p):
    """
    Calculate BO3 series win probability given single game win probability.
    Team needs to win 2 out of 3 games.
    Possible winning scenarios: WW, WLW, LWW
    """
    # Win 2-0: p * p
    # Win 2-1 (WLW): p * (1-p) * p
    # Win 2-1 (LWW): (1-p) * p * p
    bo3_prob = (p * p) + (p * (1-p) * p) + ((1-p) * p * p)
    return bo3_prob

@app.route('/api/swiss/simulate_round3_outcomes', methods=['POST'])
def simulate_round3_outcomes():
    """Simulate Round 3 match outcomes multiple times to show win distributions"""
    try:
        data = request.get_json()
        num_simulations = data.get('num_simulations', 1000)
        playin_winner = data.get('playin_winner', 'T1')
        
        worlds = initialize_tournament()
        
        # Round 3 matchups based on Round 2 state
        # High matches (2-0 vs 2-0) are BO3, others are BO1
        high_matches_bo3 = [
            ('KT Rolster', 'Top Esports'),
            ('CTBC Flying Oyster', 'Anyone s Legend')
        ]
        
        middle_matches_bo1 = [
            ('Team Secret Whales', 'FlyQuest'),
            ('Gen.G eSports', 'T1'),
            ('G2 Esports', 'Bilibili Gaming'),
            ('100 Thieves', 'Hanwha Life eSports')
        ]
        
        low_matches_bo3 = [
            ('Movistar KOI', 'Fnatic'),
            ('Vivo Keyd Stars', 'PSG Talon')
        ]
        
        # Simulate each match multiple times
        match_results = {}
        
        # High matches - BO3
        for team_a, team_b in high_matches_bo3:
            game_win_prob = worlds.get_win_probability(team_a, team_b)
            bo3_win_prob = calculate_bo3_win_probability(game_win_prob)
            
            # Simulate this match num_simulations times
            team_a_wins = sum(1 for _ in range(num_simulations) if random.random() < bo3_win_prob)
            team_b_wins = num_simulations - team_a_wins
            
            match_key = f"{team_a.lower().replace(' ', '_').replace('.', '')}_vs_{team_b.lower().replace(' ', '_').replace('.', '')}"
            match_results[match_key] = {
                'team_a': team_a,
                'team_b': team_b,
                'team_a_wins': team_a_wins,
                'team_b_wins': team_b_wins,
                'team_a_pct': (team_a_wins / num_simulations) * 100,
                'team_b_pct': (team_b_wins / num_simulations) * 100,
                'format': 'BO3'
            }
        
        # Middle matches - BO1
        for team_a, team_b in middle_matches_bo1:
            win_prob_a = worlds.get_win_probability(team_a, team_b)
            
            # Simulate this match num_simulations times
            team_a_wins = sum(1 for _ in range(num_simulations) if random.random() < win_prob_a)
            team_b_wins = num_simulations - team_a_wins
            
            match_key = f"{team_a.lower().replace(' ', '_').replace('.', '')}_vs_{team_b.lower().replace(' ', '_').replace('.', '')}"
            match_results[match_key] = {
                'team_a': team_a,
                'team_b': team_b,
                'team_a_wins': team_a_wins,
                'team_b_wins': team_b_wins,
                'team_a_pct': (team_a_wins / num_simulations) * 100,
                'team_b_pct': (team_b_wins / num_simulations) * 100,
                'format': 'BO1'
            }
        
        # Low matches - BO3
        for team_a, team_b in low_matches_bo3:
            game_win_prob = worlds.get_win_probability(team_a, team_b)
            bo3_win_prob = calculate_bo3_win_probability(game_win_prob)
            
            # Simulate this match num_simulations times
            team_a_wins = sum(1 for _ in range(num_simulations) if random.random() < bo3_win_prob)
            team_b_wins = num_simulations - team_a_wins
            
            match_key = f"{team_a.lower().replace(' ', '_').replace('.', '')}_vs_{team_b.lower().replace(' ', '_').replace('.', '')}"
            match_results[match_key] = {
                'team_a': team_a,
                'team_b': team_b,
                'team_a_wins': team_a_wins,
                'team_b_wins': team_b_wins,
                'team_a_pct': (team_a_wins / num_simulations) * 100,
                'team_b_pct': (team_b_wins / num_simulations) * 100,
                'format': 'BO3'
            }
        
        return jsonify({
            'success': True,
            'result': {
                'num_simulations': num_simulations,
                'playin_winner': playin_winner,
                'matches': match_results
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)