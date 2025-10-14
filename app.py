from flask import Flask, render_template, request, jsonify, session
import json
import random
import numpy as np
from src.tournament.worlds_tournament import WorldsTournament
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'worlds2024_simulation_key'

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
            # Multiple Swiss simulations
            all_qualified = []
            qualification_counts = {}
            
            for _ in range(num_simulations):
                swiss_teams = [
                    'Bilibili Gaming', 'Gen.G eSports', 'G2 Esports', 'FlyQuest',
                    'CTBC Flying Oyster', 'Anyone s Legend', 'Hanwha Life eSports',
                    'Movistar KOI', 'Vivo Keyd Stars', 'Team Secret Whales',
                    'KT Rolster', 'Top Esports', 'Fnatic', '100 Thieves', 'PSG Talon',
                    playin_winner
                ]
                
                qualified_teams, eliminated_teams, records, seeding, swiss_details = worlds.run_swiss_stage(swiss_teams)
                all_qualified.append(qualified_teams)
                
                for team in qualified_teams:
                    qualification_counts[team] = qualification_counts.get(team, 0) + 1
            
            # Calculate qualification statistics
            qual_stats = []
            for team, count in qualification_counts.items():
                qual_stats.append({
                    'team': team,
                    'qualifications': count,
                    'percentage': (count / num_simulations) * 100
                })
            
            qual_stats.sort(key=lambda x: x['percentage'], reverse=True)
            
            return jsonify({
                'success': True,
                'type': 'multiple',
                'result': {
                    'num_simulations': num_simulations,
                    'qualification_stats': qual_stats,
                    'playin_winner': playin_winner,
                    'seeding_strategy': seeding_strategy
                }
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
            'PCS': {'strength': 0.393, 'description': 'Struggled significantly (2-7 matches, 37.9% games, 23.5% weighted)'},
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



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)