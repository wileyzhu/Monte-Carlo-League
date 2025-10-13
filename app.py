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

@app.route('/api/generate_chart')
def generate_chart():
    """Generate championship probability chart"""
    try:
        data = request.args.get('data')
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        champion_stats = json.loads(data)
        
        # Create chart
        plt.figure(figsize=(12, 8))
        
        teams = [stat[0] for stat in champion_stats[:10]]  # Top 10
        probabilities = [stat[2] * 100 for stat in champion_stats[:10]]
        
        # Color by region
        worlds = initialize_tournament()
        colors = []
        region_colors = {
            'LCK': '#FF6B6B',
            'LPL': '#4ECDC4', 
            'LEC': '#45B7D1',
            'LTA': '#96CEB4',
            'PCS': '#FFEAA7'
        }
        
        for team in teams:
            region = worlds.team_regions.get(team, 'Unknown')
            colors.append(region_colors.get(region, '#DDA0DD'))
        
        bars = plt.bar(range(len(teams)), probabilities, color=colors)
        
        plt.title('Worlds 2024 Championship Probabilities', fontsize=16, fontweight='bold')
        plt.xlabel('Teams', fontsize=12)
        plt.ylabel('Championship Probability (%)', fontsize=12)
        plt.xticks(range(len(teams)), teams, rotation=45, ha='right')
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=color, label=region) 
                          for region, color in region_colors.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'chart': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)