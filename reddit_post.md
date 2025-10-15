# 🏆 I built a realistic Worlds 2025 Tournament Simulator with advanced statistical modeling

**TL;DR**: Created a web app that simulates League of Legends Worlds tournaments using Bayesian AR(3) models, MSI 2025 performance data, and realistic BO5 score distributions. You can run single tournaments or thousands of simulations to see championship probabilities.

## What it does

The simulator recreates the entire Worlds tournament format:
- **Play-in Stage**: T1 vs Invictus Gaming with realistic BO5 simulation (not just coin flips!)
- **Swiss Stage**: Proper 5-round format with correct seeding (5-6-5 distribution)
- **Elimination Bracket**: Single elimination with seeded matchups

## The cool technical stuff

### 🧠 Smart Probability System
Instead of random 50/50 matches, I used:
- **Bayesian AR(3) models** trained on historical performance
- **MSI 2025 + EWC 2025** regional strength adjustments
- **Game-by-game BO5 simulation** that produces realistic score distributions (3-0, 3-1, 3-2)

### 📊 Realistic Results
The BO5 simulator doesn't just pick a winner - it simulates each individual game using single-game win probabilities, so you get proper score distributions like:
- 3-0 sweeps when there's a big skill gap
- 3-2 nail-biters between evenly matched teams
- Realistic comeback potential

### 🎯 Multiple Simulation Modes
1. **Single Tournament**: See one complete tournament play out
2. **Multiple Simulations**: Run 100-1000 tournaments to get championship probabilities
3. **Real Results Input**: Enter actual tournament results and simulate from that point
4. **First Draw Analysis**: Input official Round 1 matchups and see qualification rates

## Features I'm proud of

- **Proper Swiss Stage**: Actually implements the 5-round Swiss system correctly (not the 27-round mess I had initially 😅)
- **Team Seeding**: Follows official Worlds seeding rules with Pool 1 vs Pool 3, Pool 2 vs Pool 2 matchups
- **Save/Load Functionality**: Save your custom matchup inputs
- **Regional Analysis**: See which regions are favored based on recent international performance
- **Interactive UI**: Clean Bootstrap interface with real-time results

## Tech Stack
- **Backend**: Python Flask with NumPy/Pandas for statistical modeling
- **Frontend**: Bootstrap 5 + vanilla JavaScript
- **Data**: CSV-based probability matrices from MSI/EWC performance
- **Deployment**: Ready for Heroku, Vercel, Railway, or any cloud platform

## Some interesting findings from running simulations

When I ran 1000 simulations, the championship probabilities weren't always what you'd expect from individual team strength rankings. Tournament format and bracket positioning matter a lot - sometimes a "weaker" team has better championship odds due to favorable seeding paths.

## What's next

Thinking about adding:
- Live tournament tracking during actual Worlds
- More detailed match history and statistics
- API endpoints for other developers
- Mobile-responsive improvements

The whole thing is built to be easily extensible - you could adapt it for any tournament format or esport.

---

**For the developers**: The probability matrix approach makes it super easy to plug in different data sources. The Swiss stage implementation properly handles the complex pairing logic, and the BO5 simulation uses realistic game-by-game modeling instead of just flipping coins.

**For the League fans**: It's actually fun to see how different scenarios play out. The "what if T1 makes it through play-ins vs Invictus Gaming" comparison shows some interesting differences in how the tournament could unfold.

Anyone else working on esports simulation projects? Would love to hear about different approaches to modeling team strength and tournament formats!

---

*Edit: Thanks for the gold! For those asking about the code - it's a Flask app with about 800 lines of Python for the tournament logic. The trickiest part was getting the Swiss stage pairing algorithm right and making sure the BO5 simulations felt realistic.*