"""
Tracking-Derived College Football Metrics versus NFL Rookie On-Field Performance

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import fuzzy matching
try:
    from rapidfuzz import fuzz, process
    FUZZY_AVAILABLE = True
    FUZZY_LIB = 'rapidfuzz'
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        FUZZY_AVAILABLE = True
        FUZZY_LIB = 'fuzzywuzzy'
    except ImportError:
        FUZZY_AVAILABLE = False
        FUZZY_LIB = None
        print("⚠️  No fuzzy matching library found. Install with:")
        print("   pip install rapidfuzz  (faster)")
        print("   OR pip install fuzzywuzzy python-Levenshtein")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def normalize_name(name):
    """Normalize player name for better matching."""
    if pd.isna(name):
        return ""
    name = str(name).lower().strip()
    # Remove common suffixes
    for suffix in [' jr', ' jr.', ' sr', ' sr.', ' ii', ' iii', ' iv']:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    # Remove periods and extra spaces
    name = name.replace('.', '').replace('  ', ' ')
    return name


def fuzzy_match_names(college_names, nfl_names, threshold=85):
    """
    Match college player names to NFL player names using fuzzy matching.
    
    Returns dict: {college_name: best_nfl_match} for matches above threshold
    """
    if not FUZZY_AVAILABLE:
        print("Fuzzy matching not available - using exact matching only")
        return {name: name for name in college_names if name in nfl_names}
    
    print(f"\nFuzzy matching {len(college_names)} college names to {len(nfl_names)} NFL names...")
    print(f"Using: {FUZZY_LIB} (threshold={threshold})")
    
    # Normalize all names
    college_normalized = {normalize_name(n): n for n in college_names}
    nfl_normalized = {normalize_name(n): n for n in nfl_names}
    
    matches = {}
    exact_matches = 0
    fuzzy_matches = 0
    no_match = 0
    
    for norm_college, orig_college in college_normalized.items():
        if not norm_college:
            continue
            
        # First try exact match on normalized names
        if norm_college in nfl_normalized:
            matches[orig_college] = nfl_normalized[norm_college]
            exact_matches += 1
            continue
        
        # Try fuzzy match
        result = process.extractOne(
            norm_college, 
            list(nfl_normalized.keys()),
            scorer=fuzz.ratio
        )
        
        if result:
            if FUZZY_LIB == 'rapidfuzz':
                best_match, score, _ = result
            else:
                best_match, score = result
                
            if score >= threshold:
                matches[orig_college] = nfl_normalized[best_match]
                fuzzy_matches += 1
            else:
                no_match += 1
        else:
            no_match += 1
    
    print(f"\nMatching Results:")
    print(f"  Exact matches: {exact_matches}")
    print(f"  Fuzzy matches: {fuzzy_matches}")
    print(f"  No match: {no_match}")
    print(f"  Total matched: {len(matches)} ({len(matches)/len(college_names)*100:.1f}%)")
    
    return matches


class TrackingDataAnalyzer:
    """
    Analyzer for football tracking data that processes college metrics
    and evaluates their relationship to NFL rookie performance.
    """
    
    def __init__(self, data_path=None):
        self.df = None
        self.feature_columns = []
        self.target_columns = []
        self.models = {}
        self.results = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        self.df = pd.read_csv(data_path)
        print("Dataset Loaded")
        print(f"Rows: {self.df.shape[0]}")
        print(f"Columns: {self.df.shape[1]}")
        print(f"Seasons: {self.df['season'].min()} to {self.df['season'].max()}")
        print(f"Unique players: {self.df['player_name'].nunique()}")
        print(f"Teams: {self.df['offense_team'].nunique()}")
        return self
    
    def explore_data(self):
        print("Data Exploration and Quality Check")
        
        print("\n Summary of Missing Data")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Percentage', ascending=False)
        
        if len(missing_df) > 0:
            print(f"\n{len(missing_df)} metrics have missing data:")
            print(missing_df.head(15))
            print("\nFootball Context:")
            print("Missing 'targeted' metrics means the player wasn't thrown to (limited targets)")
            print("Missing 'separation_VMAN' means the player didn't face man coverage or insufficient sample")
            print("Missing 'YACOE/CPOE' means no catch opportunities or tracking limitations")
        else:
            print("No missing data detected")
        
        print("\n\nKey Metrics Summary")
        
        key_metrics = {
            'total_plays': 'Opportunity/Sample Size',
            'max_speed_99': 'Consistent Top Speed (mph)',
            'average_separation_99': 'Separation Consistency (yards)',
            'YACOE_MEAN': 'Yards After Catch Over Expected',
            'CPOE_MEAN': 'Completion % Over Expected',
            'cod_sep_generated_overall': 'Separation from Cuts'
        }
        
        for metric, description in key_metrics.items():
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                print(f"\n{description} ({metric}):")
                print(f"  Mean: {data.mean():.2f} | Median: {data.median():.2f} | Std: {data.std():.2f}")
                print(f"  Range: [{data.min():.2f}, {data.max():.2f}]")
        
        print("\n\nPlaying Time Distribution")
        play_bins = [0, 50, 100, 150, 200, 300, 1000]
        play_labels = ['<50 (Limited)', '50-100 (Rotational)', '100-150 (Starter)', 
                       '150-200 (Heavy)', '200-300 (Workhorse)', '300+ (Elite Volume)']
        self.df['volume_tier'] = pd.cut(self.df['total_plays'], bins=play_bins, labels=play_labels)
        print(self.df['volume_tier'].value_counts().sort_index())
        
        print("\nFootball Context:")
        print("\nPlayers with low snap counts may produce unstable estimates")
        
        return self
    
    def load_rookie_nfl_performance(self, start_year=2023, end_year=2025, min_games=2):
        print("\n\nLoading True NFL Rookie Performance Data\n")

        try:
            import nfl_data_py as nfl
        except ImportError:
            raise ImportError("Please install nfl_data_py: pip install nfl_data_py")

        career = nfl.import_seasonal_data(years=range(2000, end_year + 1))
        rosters = nfl.import_seasonal_rosters(years=range(2000, end_year + 1))

        career = career.merge(
            rosters[['player_id', 'player_name', 'position']],
            on='player_id',
            how='left'
        )

        career = career[career['position'] == 'WR'].copy()

        rookie_season = (
            career.groupby('player_name')['season']
            .min()
            .reset_index()
            .rename(columns={'season': 'rookie_season'})
        )

        career = career.merge(rookie_season, on='player_name')
        rookie_wr = career[career['season'] == career['rookie_season']].copy()

        rookie_wr = rookie_wr[
            rookie_wr['rookie_season'].between(start_year, end_year)
        ]

        rookie_wr = rookie_wr[rookie_wr['games'] >= min_games].copy()
        
        # CRITICAL: Remove duplicates
        rookie_wr = rookie_wr.drop_duplicates(subset=['player_name'], keep='first')

        rookie_wr['yards_per_game'] = rookie_wr['receiving_yards'] / rookie_wr['games']
        rookie_wr['targets_per_game'] = rookie_wr['targets'] / rookie_wr['games']
        rookie_wr['receptions_per_game'] = rookie_wr['receptions'] / rookie_wr['games']
        rookie_wr['catch_rate'] = rookie_wr['receptions'] / rookie_wr['targets'].replace(0, np.nan)

        print(f"True rookie WRs loaded: {len(rookie_wr)}")
        print(f"Rookie seasons: {start_year}–{end_year}")
        print(f"Minimum games: {min_games}")

        self.rookie_perf = rookie_wr[
            [
                'player_name',
                'rookie_season',
                'yards_per_game',
                'targets_per_game',
                'receptions_per_game',
                'catch_rate'
            ]
        ].copy()

        return self
    
    def merge_tracking_with_rookie_performance(self, use_fuzzy=True, fuzzy_threshold=85):
        """
        Merge college tracking data with NFL rookie performance.
        
        Parameters:
        -----------
        use_fuzzy : bool
            Whether to use fuzzy name matching (default True)
        fuzzy_threshold : int
            Minimum fuzzy match score (0-100) to accept a match (default 85)
        """
        print("Merging College Tracking and Rookie NFL Performance")

        if not hasattr(self, 'rookie_perf'):
            raise ValueError("Rookie performance not loaded. Call load_rookie_nfl_performance() first.")

        # Keep only the most recent college season for each player
        college_final = (
            self.df.sort_values('season')
            .groupby('player_name')
            .last()
            .reset_index()
        )
        
        print(f"  College players (unique): {len(college_final)}")
        print(f"  NFL rookies available: {len(self.rookie_perf)}")

        if use_fuzzy and FUZZY_AVAILABLE:
            # Use fuzzy matching
            college_names = college_final['player_name'].unique()
            nfl_names = self.rookie_perf['player_name'].unique()
            
            name_mapping = fuzzy_match_names(college_names, nfl_names, threshold=fuzzy_threshold)
            
            # Create mapped column
            college_final['nfl_name'] = college_final['player_name'].map(name_mapping)
            
            # Filter to matched players only
            college_matched = college_final[college_final['nfl_name'].notna()].copy()
            
            # Merge using the mapped NFL name
            merged = college_matched.merge(
                self.rookie_perf,
                left_on='nfl_name',
                right_on='player_name',
                how='inner',
                suffixes=('_college', '_nfl')
            )
            
            # Clean up column names
            if 'player_name_college' in merged.columns:
                merged['player_name'] = merged['player_name_college']
                merged = merged.drop(columns=['player_name_college', 'player_name_nfl', 'nfl_name'], errors='ignore')
            
        else:
            # Fall back to exact matching
            print("  Using exact name matching")
            merged = college_final.merge(
                self.rookie_perf,
                on='player_name',
                how='inner'
            )

        # Handle season column naming
        if 'season_college' in merged.columns:
            merged['season'] = merged['season_college']
            merged = merged.drop(columns=['season_college'], errors='ignore')
        if 'season_nfl' in merged.columns:
            merged = merged.drop(columns=['season_nfl'], errors='ignore')

        # Remove any cases where college season is not before NFL rookie season
        merged = merged[merged['season'] < merged['rookie_season']]

        print(f"\nLeakage-safe players with college → rookie NFL data: {len(merged)}")

        # Verify no duplicates
        if merged['player_name'].duplicated().any():
            print("⚠️  Removing duplicate players...")
            merged = merged.drop_duplicates(subset=['player_name'], keep='first')
            print(f"  After deduplication: {len(merged)}")

        self.df = merged
        return self

    def engineer_features(self):
        print("\n\nFeature Engineering")
        
        df = self.df.copy()
        
        print("\n1. Athleticism Profile")
        df['speed_score'] = (df['max_speed_99'] + df['max_speed_30_inf_yards_max']) / 2
        print("Speed Score: Ability to reach and sustain top speed (deep threat indicator)")
        
        df['burst_rate'] = (df['high_acceleration_count_SUM'] / df['total_plays']) * 100
        print("Burst Rate: High acceleration events per play (route explosion)")
        
        df['first_step_quickness'] = df['max_speed_0_10_yards_max']
        print("First Step Quickness: Speed in first 10 yards (release & separation)")
        
        df['brake_rate'] = (df['high_deceleration_count_SUM'] / df['total_plays']) * 100
        print("Brake Rate: Deceleration events per play (cut sharpness)")
        
        print("\n\n2. Route Running Intelligence")
        df['route_diversity'] = (
            (df['10ydplus_route_MEAN'] * 0.3) +
            (df['20ydplus_route_MEAN'] * 0.4) +
            (df['changedir_route_MEAN'] * 0.3)
        )
        print("Route Diversity: Versatility across route tree (vs one-dimensional)")
        
        df['separation_consistency'] = df['average_separation_99']
        print("Separation Consistency: 99th percentile separation (elite vs lucky)")
        
        df['man_coverage_win_rate'] = df['separation_at_throw_VMAN']
        print("Man Coverage Win Rate: Separation vs man (toughest coverage)")
        
        df['tracking_skill'] = df['separation_change_postthrow_MEAN']
        print("Tracking Skill: Separation change after throw (ball tracking)")
        
        print("\n\n3. Contested Catch Ability")
        df['contested_catch_rate'] = np.where(
            df['tight_window_at_throw_SUM'] > 0,
            (df['targeted_tightwindow_catch_SUM'] / df['tight_window_at_throw_SUM']) * 100,
            np.nan
        )
        print("Contested Catch Rate: Success in tight coverage (<2 yards separation)")
        
        df['tight_window_target_pct'] = np.where(
            df['total_plays'] > 0,
            (df['tight_window_at_throw_SUM'] / df['total_plays']) * 100,
            np.nan
        )
        print("Tight Window Target %: How often QB trusts them in traffic")
        
        print("\n\n4. Playmaking and Value Creation")
        df['yac_ability'] = df['YACOE_MEAN']
        print("YAC Ability: Yards after catch over expected (playmaking)")
        
        df['qb_friendly'] = df['CPOE_MEAN']
        print("QB-Friendly Rating: Completion % over expected (reliable hands)")
        
        print("\n\n5. Change of Direction Focus")
        df['sharp_cut_ability'] = (
            df['cod_top5_speed_entry_avg_90_'] + df['cod_top5_speed_exit_avg_90_']
        ) / 2
        print("Sharp Cut Ability: Speed through 90° cuts (slants, outs, digs)")
        
        df['route_bend_ability'] = (
            df['cod_top5_speed_entry_avg_180_'] + df['cod_top5_speed_exit_avg_180_']
        ) / 2
        print("Route Bend Ability: Speed through 180° cuts (comebacks, curls)")
        
        df['cut_separation'] = df['cod_sep_generated_overall']
        print("Cut Separation: Yards of separation created from route breaks")
        
        print("\n\n6. Workload and Usage")
        df['high_volume_player'] = (df['total_plays'] >= 150).astype(int)
        print("High Volume Flag: 150+ plays (starter/feature player)")
        
        df['distance_per_play'] = df['play_distance_SUM'] / df['total_plays']
        print("Distance per Play: Route depth tendency (deep vs short game)")
        
        df['total_explosive_events'] = (
            df['high_acceleration_count_SUM'] + df['high_deceleration_count_SUM']
        )
        df['explosive_rate'] = (df['total_explosive_events'] / df['total_plays']) * 100
        print("Explosive Rate: Combined accel/decel events (dynamic playmaking)")
        
        print("\n\n7. Combined Rating Metrics")
        speed_norm = (df['speed_score'] - df['speed_score'].mean()) / df['speed_score'].std()
        burst_norm = (df['burst_rate'] - df['burst_rate'].mean()) / df['burst_rate'].std()
        df['athleticism_score'] = ((speed_norm + burst_norm) / 2) * 10 + 50
        df['athleticism_score'] = df['athleticism_score'].clip(0, 100)
        print("Athleticism Score: Combined speed + burst (0-100 scale)")
        
        route_metrics = ['route_diversity', 'separation_consistency', 'man_coverage_win_rate']
        route_cols_available = [col for col in route_metrics if col in df.columns]
        if route_cols_available:
            route_norm = df[route_cols_available].apply(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
            df['route_running_grade'] = (route_norm.mean(axis=1) * 10 + 50).clip(0, 100)
            print("Route Running Grade: Separation + versatility (0-100 scale)")
        
        key_features = ['athleticism_score', 'route_running_grade', 'yac_ability']
        available_features = [f for f in key_features if f in df.columns]
        if available_features:
            df['prospect_score'] = df[available_features].mean(axis=1)
            print("Prospect Score: Holistic evaluation (weighted composite)")
        
        print(f"Feature engineering successful: {len([c for c in df.columns if c not in self.df.columns])} new features created")
        
        self.df = df
        return self
    
    def identify_archetypes(self, n_clusters=5):
        """
        Group players into distinct archetypes using K-means clustering.
        FIXED: Uses direct assignment instead of merge to prevent row multiplication.
        """
        print("Player Archetype Identification")
        
        rows_before = len(self.df)
        
        cluster_features = ['speed_score', 'burst_rate', 'separation_consistency']
        
        available_features = []
        for feat in cluster_features:
            if feat in self.df.columns:
                non_null = self.df[feat].notna().sum()
                if non_null > 0:
                    available_features.append(feat)
                    print(f"{feat}: {non_null} non-null values")
                else:
                    print(f"{feat}: no data available")
            else:
                print(f"{feat}: column not found")
        
        if len(available_features) < 2:
            print("\nInsufficient features for clustering. Skipping archetype analysis.")
            return self
        
        # Get mask for complete data
        cluster_mask = self.df[available_features].notna().all(axis=1)
        n_complete = cluster_mask.sum()
        
        print(f"\nClustering {n_complete} players with complete data on {len(available_features)} features...")
        
        if n_complete < n_clusters:
            n_clusters = min(3, max(2, n_complete // 10))
        
        if n_complete < 10:
            print("\nToo few players for reliable clustering. Skipping.")
            self.df['archetype'] = np.nan
            return self
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.df.loc[cluster_mask, available_features])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # FIXED: Direct assignment, no merge
        self.df['archetype'] = np.nan
        self.df.loc[cluster_mask, 'archetype'] = cluster_labels
        
        print("\nReceiver Archetypes:\n")
        
        archetype_names = {
            0: "Deep Threats",
            1: "Route Technicians", 
            2: "YAC Focus",
            3: "Complete Receivers",
            4: "Possession Specialists"
        }
        
        for cluster_id in range(n_clusters):
            cluster_players = self.df[self.df['archetype'] == cluster_id]
            
            print(f"\n{archetype_names.get(cluster_id, f'Archetype {cluster_id}')} ({len(cluster_players)} players)")
            
            for feature in cluster_features:
                if feature in cluster_players.columns:
                    mean_val = cluster_players[feature].mean()
                    print(f"  {feature:30s}: {mean_val:6.2f}")
            
            sample_players = cluster_players['player_name'].head(3).tolist()
            print(f"  Example players: {', '.join(sample_players)}")
        
        # Verify row count
        rows_after = len(self.df)
        if rows_before != rows_after:
            print(f"\n⚠️  ERROR: Row count changed from {rows_before} to {rows_after}!")
        
        return self
    
    def prepare_modeling_data(self, target='targets_per_game', min_plays=50):
        print("Preparing data for modeling")

        nfl_targets = [
            'yards_per_game',
            'targets_per_game',
            'receptions_per_game',
            'catch_rate'
        ]

        if target not in nfl_targets:
            raise ValueError(
                f"Target must be a TRUE rookie NFL metric. Choose from: {nfl_targets}"
            )

        df_model = self.df.copy()
        df_model = df_model[df_model[target].notna()]

        print(f"\nPlayers with rookie NFL data ({target}): {len(df_model)}")

        df_model = df_model[df_model['total_plays'] >= min_plays]
        print(f"After min_plays filter ({min_plays}+): {len(df_model)}")

        if len(df_model) < 30:
            print("\n  Warning: Small sample size may lead to unstable results")
        
        potential_features = [
            'speed_score', 'burst_rate', 'first_step_quickness', 'brake_rate',
            'max_speed_99', 'route_diversity', 'separation_consistency', 
            'sharp_cut_ability', 'route_bend_ability', 'cut_separation',
            'changedir_route_MEAN', 'cod_sep_generated_overall',
            'distance_per_play', 'explosive_rate', 'total_plays'
        ]
        
        self.feature_columns = []
        for feat in potential_features:
            if feat in df_model.columns:
                non_null_pct = df_model[feat].notna().sum() / len(df_model) * 100
                if non_null_pct >= 50:
                    self.feature_columns.append(feat)
                    print(f"{feat:35s}: {non_null_pct:5.1f}% available")
        
        print(f"\nUsing target: {target}")
        
        if target in self.feature_columns:
            self.feature_columns.remove(target)
        
        print(f"\nFinal: {len(self.feature_columns)} features → {target}")
        
        imputer = SimpleImputer(strategy='median')
        
        X = df_model[self.feature_columns].copy()
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=self.feature_columns,
            index=X.index
        )
        
        y = df_model[target].copy()
        y = pd.Series(y.values, index=y.index, name=target)
        
        valid_idx = ~(X_imputed.isnull().any(axis=1) | y.isnull())
        X_clean = X_imputed[valid_idx]
        y_clean = y[valid_idx]
        
        print(f"\nReady: {len(X_clean)} players")
        print(f"   Target: {target}")
        print(f"   Mean: {y_clean.mean():.3f} | Median: {y_clean.median():.3f}")
        
        return X_clean, y_clean    

    def build_models(self, X, y, test_size=0.25):
        print("Building models")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\n Training set: {len(X_train)} players")
        print(f" Test set: {len(X_test)} players")
        
        print("\n Training Gradient Boosting Regressor...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='r2')
        print(f"  Cross-validation R² (5-fold): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        y_pred_train = gb_model.predict(X_train)
        y_pred_test = gb_model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\n Model Performance:")
        print(f"  Training R²: {train_r2:.3f}")
        print(f"  Test R²: {test_r2:.3f}")
        print(f"  Test MAE: {test_mae:.3f}")
        print(f"  Test RMSE: {test_rmse:.3f}")
        
        self.models['gradient_boosting'] = gb_model
        self.results['predictions'] = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test
        }
        self.results['metrics'] = {
            'train_r2': train_r2, 'test_r2': test_r2,
            'test_mae': test_mae, 'test_rmse': test_rmse,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
        }
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': gb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.results['feature_importance'] = feature_importance
        
        print("\n Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return self


if __name__ == "__main__":
    print("""
        College Tracking Data to NFL Rookie On-Field Performance
        WITH FUZZY NAME MATCHING                    
    """)
    
    analyzer = TrackingDataAnalyzer()
    
    print("\nRun the full pipeline:")
    print("   analyzer.load_data('your_data.csv')")
    print("   analyzer.explore_data()")
    print("   analyzer.engineer_features()")
    print("   analyzer.load_rookie_nfl_performance()")
    print("   analyzer.merge_tracking_with_rookie_performance(use_fuzzy=True)")
    print("   analyzer.identify_archetypes()")
    print("   X, y = analyzer.prepare_modeling_data()")
    print("   analyzer.build_models(X, y)")