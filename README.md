# 🏆 Worlds 2025 Tournament Simulator

A realistic League of Legends World Championship simulator using advanced statistical modeling and MSI 2025 performance data.

## 🚀 Quick Start

### Local Development
```bash
git clone <repository-url>
cd Monte_Carlo
pip install -r requirements.txt
python app.py
```
Open http://localhost:3000

### Deploy to Railway (Recommended)
1. Go to https://railway.app
2. Connect your GitHub repository  
3. Deploy automatically - no configuration needed!

## 📁 Project Structure

```
├── src/                          # Core application code
│   ├── data/                     # Data processing modules
│   │   ├── data_collection.py    # Data gathering utilities
│   │   ├── data_preprocessing.py # Data cleaning and preparation
│   │   ├── regional_adjustment.py # Regional performance adjustments
│   │   └── msi_regional_analysis.py # Regional analysis tools
│   ├── tournament/               # Tournament logic
│   │   ├── elimination.py        # Elimination bracket logic
│   │   ├── swiss_stage.py       # Swiss format implementation
│   │   └── worlds_tournament.py # Main tournament orchestration
│   └── models/                   # Prediction models
│       ├── probability_matrix.py # Team matchup probabilities
│       ├── load_and_predict.py  # Model loading and prediction
│       └── state_space.py       # State space modeling
├── static/                       # Web assets (CSS, JS)
├── templates/                    # HTML templates
├── scripts/                      # Utility scripts
├── tests/                        # Test modules
├── dataset/                      # Data files
├── models/                       # Trained model files
└── app.py                        # Main Flask application
```

## ⚡ Features

### Tournament Formats
- **Swiss Stage**: 16 teams, 3 wins to advance, 3 losses eliminated
- **Elimination Brackets**: Single/double elimination with seeding
- **Play-in Stage**: Qualification rounds for lower-seeded teams

### Prediction Models
- **Bayesian AR(3) Model**: Historical performance-based predictions
- **Regional Adjustments**: MSI-based inter-regional strength calibration
- **Monte Carlo Simulation**: Statistical analysis through multiple runs

### Web Interface
- Interactive bracket visualization
- Real-time simulation results
- Team performance analytics
- Tournament progression tracking

## 🎯 Usage Examples

### Generate Probability Matrix
```bash
python src/models/probability_matrix.py
```

### Apply Regional Adjustments
```bash
python src/data/regional_adjustment.py
```

### Run Tests
```bash
python -m pytest tests/
```

## 📊 Regional Strength Analysis

Based on MSI performance data:
- **LCK (Korea)**: 1.000 - Dominant regional performance
- **LPL (China)**: 0.808 - Strong secondary region
- **LCP (Taiwan/Vietnam)**: 0.510 - Emerging region strength
- **LTA (Americas)**: 0.487 - Competitive but inconsistent
- **LEC (Europe)**: 0.400 - Rebuilding phase

## 🛠️ Technical Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Models**: Bayesian AR(3), State Space Models
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Chart.js
- **Testing**: Pytest

## 📈 Model Performance

The system uses empirically-derived regional adjustments from international tournament results, providing:
- Accurate inter-regional matchup predictions
- Statistical confidence intervals
- Historical performance validation
- Monte Carlo simulation reliability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes in the appropriate `src/` directory
4. Add tests in the `tests/` directory
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.