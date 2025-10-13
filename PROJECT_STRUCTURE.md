# Project Structure

## Tournament Prediction System

```
Monte_Carlo/
├── src/                          # Core application code
│   ├── data/                     # Data processing modules
│   │   ├── data_collection.py    # Data gathering utilities
│   │   ├── data_preprocessing.py # Data cleaning and preparation
│   │   ├── regional_adjustment.py # Regional performance adjustments
│   │   ├── regional_adjustment_msi.py # MSI regional adjustments
│   │   └── msi_regional_analysis.py # MSI analysis tools
│   ├── tournament/               # Tournament logic
│   │   ├── elimination.py        # Elimination bracket logic
│   │   ├── playin.py            # Play-in stage management
│   │   ├── swiss_stage.py       # Swiss format implementation
│   │   └── worlds_tournament.py # Main tournament orchestration
│   └── models/                   # Prediction models
│       ├── probability_matrix.py # Team matchup probabilities
│       ├── load_and_predict.py  # Model loading and prediction
│       └── state_space.py       # State space modeling
├── static/                       # Web assets
│   ├── css/                     # Stylesheets
│   └── js/                      # JavaScript files
├── templates/                    # HTML templates
│   ├── index.html               # Main web interface
│   └── test_bracket.html        # Bracket testing page
├── scripts/                      # Utility scripts
│   ├── setup_webapp.py          # Web app setup
│   └── run_webapp.py            # Web app runner
├── tests/                        # Test modules
│   └── test_ar_model.py         # Model testing
├── dataset/                      # Data files
├── models/                       # Trained model files
├── app.py                        # Main Flask application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Quick Start

1. **Run the web application:**
   ```bash
   python app.py
   ```

2. **Setup web app (if needed):**
   ```bash
   python scripts/setup_webapp.py
   ```

3. **Run tests:**
   ```bash
   python -m pytest tests/
   ```