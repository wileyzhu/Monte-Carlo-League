"""
Model Loading and Prediction Module
Loads datasets, applies preprocessing, loads trained models, and generates predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import data_preprocessing

# Import dependencies with availability checks
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available")

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available")

def load_datasets():
    """Load and concatenate the example match datasets"""
    dataset_path = Path("dataset")
    
    # Load both datasets
    df1 = pd.read_csv(dataset_path / "example_matches.csv")
    df2 = pd.read_csv(dataset_path / "example_matches1.csv")
    
    # Concatenate datasets
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df

def load_best_models():
    """Load the best trained models for each role"""
    models_path = Path("models")
    roles = ['top', 'jungle', 'mid', 'adc', 'support']
    models = {}
    
    for role in roles:
        model_loaded = False
        
        # Try TensorFlow models first (.h5 extension)
        if TF_AVAILABLE:
            h5_file = models_path / f"best_{role}_model.h5"
            if h5_file.exists():
                try:
                    models[role] = load_model(str(h5_file), compile=False)
                    print(f"Loaded best_{role}_model.h5")
                    model_loaded = True
                except Exception as e:
                    print(f"Error loading {role} .h5 model: {e}")
        
        # Try joblib models (.pkl extension) as fallback
        if not model_loaded and JOBLIB_AVAILABLE:
            pkl_file = models_path / f"best_{role}_model.pkl"
            if pkl_file.exists():
                try:
                    models[role] = joblib.load(str(pkl_file))
                    print(f"Loaded best_{role}_model.pkl")
                    model_loaded = True
                except Exception as e:
                    print(f"Error loading {role} .pkl model: {e}")
        
        if not model_loaded:
            print(f"Warning: No model found for {role}")
    
    return models


def add_predictions(df, models):
    """Generate role-specific predictions and add to dataframe"""
    roles = ['top', 'jungle', 'mid', 'adc', 'support']
    
    # Initialize prediction columns
    for role in roles:
        df[f'{role}_prediction'] = np.nan
    
    for role in roles:
        if role in models and models[role] is not None:
            try:
                # Filter for role-specific players
                role_mask = df['Role'].str.lower() == role.lower()
                role_data = df[role_mask]
                
                if len(role_data) > 0:
                    # Process data once using the preprocessing pipeline
                    processor = data_preprocessing.DataProcessor()
                    df_engineered = processor.engineer_features(role_data)
                    role_datasets = processor.prepare_role_datasets(df_engineered)
                    
                    if role.upper() in role_datasets:
                        clean_role_data = role_datasets[role.upper()]
                        
                        # Apply scaling using preprocess_split
                        X_scaled, feature_names = processor.preprocess_split(clean_role_data)
                        
                        if X_scaled.shape[0] > 0:
                            # Generate predictions
                            predictions = models[role].predict(X_scaled, verbose=0)
                            predictions_flat = predictions.flatten() if predictions.ndim > 1 else predictions
                            
                            # Assign predictions to players with complete data
                            if len(predictions_flat) == len(clean_role_data):
                                df.loc[clean_role_data.index, f'{role}_prediction'] = predictions_flat
                                print(f"Added {role} predictions for {len(clean_role_data)} players")
                            else:
                                print(f"Shape mismatch for {role}: {len(predictions_flat)} vs {len(clean_role_data)}")
                        else:
                            print(f"No valid features for {role}")
                    else:
                        print(f"No {role} dataset found")
                else:
                    print(f"No {role} players found")
                    
            except Exception as e:
                print(f"Error making predictions for {role}: {e}")
        else:
            print(f"No model available for {role}")
    
    return df

def main():
    """Main pipeline: load data, preprocess, load models, and generate predictions"""
    # Load and combine datasets
    print("Loading datasets...")
    df = load_datasets()
    
    # Apply feature engineering
    print("\nPreprocessing data...")
    processor = data_preprocessing.DataProcessor()
    df_processed = processor.engineer_features(df)
    print(f"Data after preprocessing: {df_processed.shape}")
    
    # Load trained models
    print("\nLoading best models...")
    models = load_best_models()
    
    # Generate predictions
    print("\nMaking predictions...")
    df_with_predictions = add_predictions(df_processed, models)
    
    # Save results
    print("\nSaving results...")
    output_path = "dataset/matches_with_predictions.csv"
    df_with_predictions.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    
    # Display summary
    print(f"\nFinal dataset shape: {df_with_predictions.shape}")
    pred_cols = [col for col in df_with_predictions.columns if 'prediction' in col]
    if pred_cols:
        print(f"\nPrediction summary:")
        print(df_with_predictions[pred_cols].describe())
    
    return df_with_predictions

if __name__ == "__main__":
    df_final = main()