"""
Master runner for WR tracking → NFL rookie performance analysis.
Orchestrates the full pipeline from data load to results export.
"""

import os
import sys
import argparse
from datetime import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Try to import from src/ package structure first, fall back to local
try:
    from src.tracking_analysis_nfl import TrackingDataAnalyzer
    from src.advanced_modeling import AdvancedModelingPipeline
    from src.tracking_visuals import create_all_visuals
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    try:
        from tracking_analysis_nfl import TrackingDataAnalyzer
        from advanced_modeling import AdvancedModelingPipeline
        from tracking_visuals import create_all_visuals
    except ImportError as e:
        print(f"Import error: {e}")
        sys.exit(1)


def show_banner():
    """ print ASCII header """
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║            WR TRACKING DATA → NFL ROOKIE PERFORMANCE                      ║
║                    WITH FUZZY NAME MATCHING                               ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")
    print(f"Started: {datetime.now().strftime('%B %d, %Y %I:%M %p')}\n")


def setup_dirs(base):
    """Create output folder structure - keeps things organized."""
    paths = {
        "viz": os.path.join(base, "visualizations"),
        "rpt": os.path.join(base, "reports"),
        "dat": os.path.join(base, "data"),
        "mdl": os.path.join(base, "models"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def get_feature_importance(mdl, X):
    """Extract feature importance from whatever model type we have."""
    # Tree models have feature_importances_, linear models have coef_
    if hasattr(mdl, "feature_importances_"):
        imp = mdl.feature_importances_
    elif hasattr(mdl, "coef_"):
        imp = np.abs(mdl.coef_)
    else:
        # Fallback: equal weights 
        imp = np.ones(X.shape[1]) / X.shape[1]

    return (
        pd.DataFrame({"feature": X.columns, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to tracking CSV")
    parser.add_argument("--output", default="results")
    parser.add_argument("--min-plays", type=int, default=50)
    parser.add_argument("--quick", action="store_true", help="Skip ensemble/SHAP/uncertainty (faster)")
    parser.add_argument("--rookie-start-year", type=int, default=2023)
    parser.add_argument("--rookie-end-year", type=int, default=2024)
    parser.add_argument("--target", type=str, default="targets_per_game",
                        choices=["targets_per_game", "yards_per_game", "receptions_per_game", "catch_rate"])
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable fuzzy name matching")
    parser.add_argument("--fuzzy-threshold", type=int, default=85, 
                        help="Fuzzy match threshold (0-100, default 85)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"\n✗ Data file not found: {args.data}")
        sys.exit(1)

    show_banner()
    dirs = setup_dirs(args.output)

    # Load tracking data and engineer features
    analyzer = TrackingDataAnalyzer(args.data)
    analyzer.explore_data()
    analyzer.engineer_features()

    # Pull NFL rookie stats and match to college players
    analyzer.load_rookie_nfl_performance(
        start_year=args.rookie_start_year,
        end_year=args.rookie_end_year
    )
    
    # Fuzzy matching helps catch name variations (e.g., "Chris Olave" vs "Christopher Olave")
    use_fuzzy = not args.no_fuzzy
    analyzer.merge_tracking_with_rookie_performance(
        use_fuzzy=use_fuzzy,
        fuzzy_threshold=args.fuzzy_threshold
    )
    
    # Quick sanity check on data availability
    print(f"\nDEBUG: df has {len(analyzer.df)} rows")
    print(f"DEBUG: {args.target} available for {analyzer.df[args.target].notna().sum()} players")
    
    # Cluster players into archetypes (speed guys, route runners, etc.)
    analyzer.identify_archetypes(n_clusters=5)

    # Prep clean X/y for modeling
    X, y = analyzer.prepare_modeling_data(
        target=args.target,
        min_plays=args.min_plays
    )

    if X is None or len(X) < 10:
        print("\n✗ Not enough data to build models")
        sys.exit(1)

    # Build and compare multiple model types
    pipe = AdvancedModelingPipeline()
    pipe.build_model_suite()
    pipe.compare_models(X, y)

    best = pipe.best_model

    # Full analysis takes longer but gives better insights
    if not args.quick:
        pipe.build_ensemble(X, y)
        pipe.analyze_feature_interactions(X, y)
        
        # Tune hyperparams for XGB/LGB if they won
        winner = pipe.results['model_comparison'].iloc[0]['Model']
        if winner in ['xgboost', 'lightgbm']:
            pipe.hyperparameter_tuning(X, y, model_name=winner)
            pipe.compare_models(X, y)  # Re-run comparison with tuned version
        
        # SHAP values for interpretability
        try:
            pipe.shap_analysis(X, sample_size=min(100, len(X)))
        except Exception as e:
            print(f"SHAP skipped: {str(e)}")
        
        # Bootstrap for confidence intervals
        pipe.quantify_uncertainty(X, y, n_iterations=50)

    # Compute final predictions and metrics for export
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    y_pred_tr = best.predict(X_tr)
    y_pred_te = best.predict(X_te)

    tgt_col = args.target
    if hasattr(y, 'name') and y.name:
        tgt_col = y.name

    # Package results for visualization and export
    analyzer.results = {
        "model_comparison": pipe.results["model_comparison"],
        "feature_importance": get_feature_importance(best, X),
        "predictions": {
            "X_train": X_tr,
            "X_test": X_te,
            "y_train": y_tr,
            "y_pred_train": y_pred_tr,
            "y_test": y_te,
            "y_pred_test": y_pred_te,
            "target_name": tgt_col,
        },
        "metrics": {
            "train_r2": r2_score(y_tr, y_pred_tr),
            "test_r2": r2_score(y_te, y_pred_te),
            "train_mae": mean_absolute_error(y_tr, y_pred_tr),
            "test_mae": mean_absolute_error(y_te, y_pred_te),
            "train_rmse": np.sqrt(mean_squared_error(y_tr, y_pred_tr)),
            "test_rmse": np.sqrt(mean_squared_error(y_te, y_pred_te)),
            "cv_mean": pipe.results["model_comparison"].iloc[0]['CV R² Mean'] if 'CV R² Mean' in pipe.results["model_comparison"].columns else 0,
            "cv_std": pipe.results["model_comparison"].iloc[0]['CV R² Std'] if 'CV R² Std' in pipe.results["model_comparison"].columns else 0,
        },
    }

    analyzer.models = pipe.models

    # Generate all plots
    create_all_visuals(analyzer, output_dir=dirs["viz"])

    # Export processed data and results
    analyzer.df.to_csv(
        os.path.join(dirs["dat"], "processed_data_with_features.csv"),
        index=False,
    )

    pipe.results["model_comparison"].to_csv(
        os.path.join(dirs["rpt"], "model_comparison.csv"),
        index=False,
    )

    winner_row = pipe.results["model_comparison"].iloc[0]
    
    # Pickle export for Streamlit dashboard integration
    import pickle

    export_pkg = {
        'processed_data': analyzer.df,
        'feature_importance': analyzer.results.get('feature_importance', pd.DataFrame()),
        'predictions': analyzer.results.get('predictions', {}),
        'metrics': analyzer.results.get('metrics', {}),
        'model_comparison': analyzer.results.get('model_comparison', pd.DataFrame()),
        'analysis_type': 'nfl_rookie_performance',
        'target_variable': tgt_col,
        'sample_size': len(analyzer.df),
        'n_features': len(X.columns) if X is not None else 0
    }
    with open('tracking_nfl_export.pkl', 'wb') as f:
        pickle.dump(export_pkg, f)

    # Print summary with context
    print("\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
    print(f"\nSample: {len(X)} players (college → NFL matched)")
    print(f"Target: {tgt_col}")
    print(f"Best Model: {winner_row['Model'].upper()}")
    print(f"Test R²: {winner_row['Test R²']:.3f}")
    print(f"Test MAE: {winner_row['Test MAE']:.3f}")
    print(f"CV R²: {winner_row['CV R² Mean']:.3f}")
    
    # Context matters - combine metrics tell us almost nothing (R² = -0.155)
    print(f"\n--- WHAT THIS MEANS ---")
    if winner_row['Test R²'] > 0.15:
        print(f"→ Model explains {winner_row['Test R²']*100:.1f}% of NFL rookie variance")
        print(f"→ This BEATS combine testing (R² = -0.155)")
        print(f"→ Tracking data captures real game skills")
    elif winner_row['Test R²'] > 0:
        print(f"→ Model explains {winner_row['Test R²']*100:.1f}% of variance (modest)")
        print(f"→ Still better than combine (R² = -0.155)")
        print(f"→ NFL success is hard to predict, but this helps")
    else:
        print(f"→ Model R² = {winner_row['Test R²']:.3f} (below baseline)")
        print(f"→ NFL rookie performance is inherently noisy")
        print(f"→ Many factors beyond college metrics matter")
    
    print(f"\n✓ Results saved to: {args.output}/")


if __name__ == "__main__":
    main()