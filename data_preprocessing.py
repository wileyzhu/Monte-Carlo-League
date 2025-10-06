"""
Data Preprocessing Module for League of Legends Match Analysis
Handles data loading, merging, feature engineering, and dataset preparation
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


class DataProcessor:
    """Handles all data preprocessing tasks for LoL match analysis"""
    
    def __init__(self):
        self.df = None
        self.processed_df = None

    # -----------------------------------------------------
    # 1. DATA LOADING AND MERGING
    # -----------------------------------------------------
    def load_and_merge_batches(self, base_path: str, num_batches: int = 11) -> pd.DataFrame:
        print("Loading and merging batch files...")
        df = pd.read_csv(f"{base_path}/all_matches.csv")
        print(f"Loaded main file: {len(df)} records")
        
        for i in range(num_batches):
            try:
                batch_df = pd.read_csv(f"{base_path}/all_matches_batch_{i}.csv")
                df = pd.concat([df, batch_df], ignore_index=True)
                print(f"Merged batch {i}: {len(batch_df)} records")
            except FileNotFoundError:
                print(f"Batch file {i} not found, skipping...")
        
        df.to_csv(f"{base_path}/all_matches_final.csv", index=False)
        print(f"Final merged dataset: {len(df)} records")
        
        self.df = df
        return df

    # -----------------------------------------------------
    # 2. SERIES FLAGGING
    # -----------------------------------------------------
    def assign_bo5_series(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df.copy()
        print("Identifying Bo5 series and clutch games...")

        game_df = df.drop_duplicates(subset="Game_ID").sort_values(by="Game_ID").reset_index(drop=True)
        bo5_flags = [False] * len(game_df)
        final3_game = [False] * len(game_df)
        series_start = 0
        
        for i in range(1, len(game_df)):
            teams_now = {game_df.loc[i, 'Blue_Team'], game_df.loc[i, 'Red_Team']}
            teams_prev = {game_df.loc[i - 1, 'Blue_Team'], game_df.loc[i - 1, 'Red_Team']}
            if teams_now != teams_prev:
                series_len = i - series_start
                if (series_len == 3 and
                    game_df.loc[series_start, 'Winning_Team'] == game_df.loc[series_start + 1, 'Winning_Team']):
                    for j in range(series_start, i): bo5_flags[j] = True
                elif series_len >= 4:
                    for j in range(series_start, i): bo5_flags[j] = True
                    for j in range(i - 3, i): final3_game[j] = True
                series_start = i
        
        df = df.merge(
            game_df.assign(bo5=bo5_flags, final3_game=final3_game)[['Game_ID', 'bo5', 'final3_game']],
            on='Game_ID', how='left'
        )
        print(f"Identified {sum(bo5_flags)} Bo5 games, {sum(final3_game)} clutch games")
        return df

    # -----------------------------------------------------
    # 3. FEATURE ENGINEERING
    # -----------------------------------------------------
    def engineer_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df.copy()
        print("Engineering features...")

        df['KDE'] = (df['Kills'] + df['Assists']) / (df['Deaths'] + 1)
        df['Multi-Kill'] = np.where(
            (df['Double kills'] > 0) | (df['Triple kills'] > 0) |
            (df['Quadra kills'] > 0) | (df['Penta kills'] > 0), 1, 0)
        df['DTPD'] = df['Total damage taken'] / (df['Deaths'] + 1)
        for col in ['DMG%', 'KP%']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip('%').astype(float) / 100
        print("Feature engineering done.")
        return df

    # -----------------------------------------------------
    # 4. SPLIT ROLE DATASETS
    # -----------------------------------------------------
    def prepare_role_datasets(self, df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        if df is None:
            df = self.df.copy()
        print("Preparing role-specific datasets...")

        selected_features = [
            'KDE', 'DPM', 'Multi-Kill', 'GPM', 'VSPM', 'WCPM',
            'GD@15', 'XPD@15', 'CSD@15', 'LVLD@15', 'DTPD'
        ]
        available = [f for f in selected_features if f in df.columns]
        df_clean = df[available + ['Win', 'Role', 'Game_ID']].copy()
        roles = ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT']
        role_datasets = {}

        for role in roles:
            if 'Role' not in df_clean.columns:
                raise ValueError("Missing 'Role' column in dataset.")
            role_data = df_clean[df_clean['Role'] == role].dropna(subset=available)
            role_datasets[role] = role_data
            print(f"{role}: {len(role_data)} samples")
        return role_datasets

    # -----------------------------------------------------
    # 5. TRAIN/TEST SPLIT + SCALING
    # -----------------------------------------------------
    def preprocess_split(self, df: pd.DataFrame):
        X = df.drop(columns=['Win', 'Game_ID'])
        y = df['Win']

        mean_cols = ['GD@15', 'XPD@15', 'CSD@15']
        median_cols = ['VSPM', 'DTPD']
        categorical = ['Role']  # Multi-Kill is binary, not categorical

        # Drop only truly categorical columns (Role), keep Multi-Kill
        X = X.drop(columns=[col for col in categorical if col in X.columns])

        mean_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        median_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
        scale_pipe = Pipeline([('scaler', StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[
                ('mean', mean_pipe, [c for c in mean_cols if c in X.columns]),
                ('median', median_pipe, [c for c in median_cols if c in X.columns]),
                ('scale', scale_pipe, [
                    c for c in X.columns if c not in mean_cols + median_cols
                ])
            ],
            remainder='drop', verbose_feature_names_out=False
        )

        X_preprocessed = preprocessor.fit_transform(X)
        feature_names = preprocessor.get_feature_names_out()
        return X_preprocessed, feature_names

    # -----------------------------------------------------
    # 6. MASTER PIPELINE
    # -----------------------------------------------------
    def process_full_pipeline(self, base_path: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Tuple]]:
        print("Running full preprocessing pipeline...")
        df = self.load_and_merge_batches(base_path)
        df = self.assign_bo5_series(df)
        df = self.engineer_features(df)
        role_datasets = self.prepare_role_datasets(df)

        processed_roles = {}
        for role, data in role_datasets.items():
            X, feature_names = self.preprocess_split(data)
            processed_roles[role] = X

        self.processed_df = df
        print("✅ Full pipeline completed successfully.")
        return df, role_datasets, processed_roles
