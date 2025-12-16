"""
Advanced Multi-Model Analysis Pipeline
======================================

This module demonstrates sophisticated modeling techniques while maintaining
interpretability - the sweet spot for sports analytics applications.

Key Features:
- Multiple model comparison (shows you understand trade-offs)
- Ensemble methods (combines strengths)
- Feature interaction analysis (uncovers hidden patterns)
- Uncertainty quantification (honest about confidence)
- SHAP values (modern interpretability)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Optional: SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️  SHAP not installed. Install with: pip install shap")


class AdvancedModelingPipeline:
    """
    Sophisticated multi-model pipeline that demonstrates technical depth.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.ensemble_weights = {}
        
    def build_model_suite(self):
        """
        Create a suite of models with clear rationale for each.
        """
        print("\n" + "="*80)
        print("BUILDING MODEL SUITE WITH STRATEGIC RATIONALE")
        print("="*80)
        
        models = {
            # BASELINE: Simple interpretable model
            'ridge': {
                'model': Ridge(alpha=1.0),
                'rationale': 'Baseline linear model - interpretable coefficients',
                'strength': 'Clear feature relationships',
                'weakness': 'Assumes linearity'
            },
            
            # TREE ENSEMBLE: Capture non-linearity
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=-1
                ),
                'rationale': 'Handles non-linearity, robust to outliers',
                'strength': 'Feature interactions, overfitting resistant',
                'weakness': 'Can miss subtle patterns'
            },
            
            # GRADIENT BOOSTING: Sequential learning
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    subsample=0.8,
                    random_state=42
                ),
                'rationale': 'Sequential error correction, strong performance',
                'strength': 'Captures complex patterns iteratively',
                'weakness': 'Slower training than RF'
            },
            
            # XGBOOST: Optimized gradient boosting
            'xgboost': {
                'model': xgb.XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    min_child_weight=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1
                ),
                'rationale': 'Regularized boosting, prevents overfitting',
                'strength': 'Built-in regularization, faster than sklearn GB',
                'weakness': 'More hyperparameters to tune'
            },
            
            # LIGHTGBM: Efficient gradient boosting
            'lightgbm': {
                'model': lgb.LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=5,
                    num_leaves=31,
                    min_child_samples=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'rationale': 'Fast histogram-based boosting, handles large data',
                'strength': 'Extremely fast, memory efficient',
                'weakness': 'Less interpretable than simpler models'
            },
            
            # ELASTIC NET: Regularized linear with feature selection
            'elastic_net': {
                'model': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
                'rationale': 'Feature selection via L1, stability via L2',
                'strength': 'Automatic feature selection',
                'weakness': 'Linear assumptions'
            }
        }
        
        print("\n📊 MODEL SUITE COMPOSITION:\n")
        for name, info in models.items():
            print(f"{name.upper()}")
            print(f"  Rationale: {info['rationale']}")
            print(f"  ✓ Strength: {info['strength']}")
            print(f"  ✗ Weakness: {info['weakness']}\n")
        
        self.models = {name: info['model'] for name, info in models.items()}
        return self
    
    def compare_models(self, X, y, cv_folds=5):
        """
        Rigorous model comparison with multiple metrics.
        
        This is what separates good analysts from great ones:
        - Not just "which performs best?"
        - But "what are the trade-offs?"
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        print(f"\nValidation Strategy: {cv_folds}-Fold Cross-Validation")
        print(f"Training Samples: {len(X):,}")
        print(f"Features: {X.shape[1]}\n")
        
        from sklearn.model_selection import train_test_split
        
        # Hold out test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        results = []
        
        for name, model in self.models.items():
            print(f"\n🤖 Training {name.upper()}...")
            
            # Cross-validation on training set
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv_folds, 
                scoring='r2',
                n_jobs=-1
            )
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Overfitting check
            overfit_gap = train_r2 - test_r2
            
            results.append({
                'Model': name,
                'CV R² Mean': cv_scores.mean(),
                'CV R² Std': cv_scores.std(),
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Test MAE': test_mae,
                'Test RMSE': test_rmse,
                'Overfit Gap': overfit_gap
            })
            
            print(f"  CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  Test R²: {test_r2:.3f} | MAE: {test_mae:.3f}")
            print(f"  Overfit Gap: {overfit_gap:.3f}", 
                  "⚠️ HIGH" if overfit_gap > 0.15 else "✓ Good")
        
        # Results summary
        results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
        
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Identify best model
        best_idx = results_df['Test R²'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 BEST MODEL: {best_model_name.upper()}")
        print(f"   Test R²: {results_df.loc[best_idx, 'Test R²']:.3f}")
        print(f"   Test MAE: {results_df.loc[best_idx, 'Test MAE']:.3f}")
        
        self.results['model_comparison'] = results_df
        self.results['X_train'] = X_train
        self.results['X_test'] = X_test
        self.results['y_train'] = y_train
        self.results['y_test'] = y_test
        
        return self
    
    def hyperparameter_tuning(self, X, y, model_name='xgboost'):
        """
        Demonstrate proper hyperparameter optimization.
        
        Shows you understand:
        - Grid search vs random search
        - Cross-validation for tuning
        - Bias-variance trade-off
        """
        print("\n" + "="*80)
        print(f"HYPERPARAMETER TUNING: {model_name.upper()}")
        print("="*80)
        
        if model_name == 'xgboost':
            param_grid = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
        elif model_name == 'lightgbm':
            param_grid = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'num_leaves': [15, 31, 63],
                'min_child_samples': [5, 10, 20]
            }
            base_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        
        else:
            print(f"⚠️  Tuning not implemented for {model_name}")
            return self
        
        print(f"\n🔍 Searching {len(param_grid)} hyperparameters...")
        print(f"   Search space: ~{np.prod([len(v) for v in param_grid.values()]):,} combinations")
        
        # Use RandomizedSearchCV for efficiency
        from sklearn.model_selection import RandomizedSearchCV
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=50,  # Sample 50 combinations
            scoring='r2',
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(X, y)
        
        print(f"\n✅ TUNING COMPLETE")
        print(f"   Best CV R²: {search.best_score_:.3f}")
        print(f"   Best Parameters:")
        for param, value in search.best_params_.items():
            print(f"      {param}: {value}")
        
        self.models[f'{model_name}_tuned'] = search.best_estimator_
        self.results[f'{model_name}_tuning'] = {
            'best_score': search.best_score_,
            'best_params': search.best_params_
        }
        
        return self
    
    def build_ensemble(self, X, y, top_n=3):
        """
        Create weighted ensemble of best models.
        
        Sophisticated technique that shows:
        - Understanding model complementarity
        - Weighted averaging based on performance
        - Ensemble learning principles
        """
        print("\n" + "="*80)
        print("BUILDING MODEL ENSEMBLE")
        print("="*80)
        
        if 'model_comparison' not in self.results:
            print("⚠️  Run compare_models() first!")
            return self
        
        # Get top N models
        top_models = self.results['model_comparison'].head(top_n)
        
        print(f"\n🎯 Ensemble Components (Top {top_n} by Test R²):\n")
        print(top_models[['Model', 'Test R²', 'Test MAE']].to_string(index=False))
        
        # Get predictions from each model
        X_test = self.results['X_test']
        y_test = self.results['y_test']
        
        predictions = {}
        for _, row in top_models.iterrows():
            model_name = row['Model']
            model = self.models[model_name]
            predictions[model_name] = model.predict(X_test)
        
        # Weight by inverse MAE (better models get higher weight)
        weights = {}
        total_inv_mae = 0
        for _, row in top_models.iterrows():
            inv_mae = 1 / row['Test MAE']
            weights[row['Model']] = inv_mae
            total_inv_mae += inv_mae
        
        # Normalize weights
        for model_name in weights:
            weights[model_name] /= total_inv_mae
        
        print("\n📊 Ensemble Weights:")
        for model_name, weight in weights.items():
            print(f"   {model_name:20s}: {weight:.3f}")
        
        # Weighted average prediction
        ensemble_pred = np.zeros(len(y_test))
        for model_name, pred in predictions.items():
            ensemble_pred += pred * weights[model_name]
        
        # Evaluate ensemble
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        
        print(f"\n🏆 ENSEMBLE PERFORMANCE:")
        print(f"   Test R²: {ensemble_r2:.3f}")
        print(f"   Test MAE: {ensemble_mae:.3f}")
        
        # Compare to best single model
        best_single_r2 = top_models.iloc[0]['Test R²']
        improvement = ensemble_r2 - best_single_r2
        
        print(f"\n   vs Best Single Model:")
        print(f"   Improvement: {improvement:+.3f} R²")
        print(f"   {'✓ Ensemble wins!' if improvement > 0 else '⚠️ Single model better (use that)'}")
        
        self.ensemble_weights = weights
        self.results['ensemble'] = {
            'predictions': ensemble_pred,
            'r2': ensemble_r2,
            'mae': ensemble_mae,
            'weights': weights
        }
        
        return self
    
    def analyze_feature_interactions(self, X, y, top_n_features=10):
        """
        Identify important feature interactions.
        
        Advanced technique that reveals:
        - Speed + Separation interaction (multiplicative effect)
        - Route diversity × YAC (versatile playmakers)
        - Volume × Efficiency (durability insights)
        """
        print("\n" + "="*80)
        print("FEATURE INTERACTION ANALYSIS")
        print("="*80)
        
        print("\n🔍 Analyzing pairwise feature interactions...")
        
        # Get feature importance from best model
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            top_features = X.columns[np.argsort(importance)[-top_n_features:]]
        else:
            # Use correlation with target as fallback
            correlations = X.corrwith(y).abs()
            top_features = correlations.nlargest(top_n_features).index
        
        print(f"\n📊 Testing interactions among top {len(top_features)} features:")
        for feat in top_features:
            print(f"   • {feat}")
        
        # Create interaction features
        X_with_interactions = X.copy()
        interaction_names = []
        
        from itertools import combinations
        for feat1, feat2 in combinations(top_features, 2):
            interaction_name = f"{feat1}×{feat2}"
            X_with_interactions[interaction_name] = X[feat1] * X[feat2]
            interaction_names.append(interaction_name)
        
        # Train model with interactions
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_interactions, y, test_size=0.25, random_state=42
        )
        
        # Use a simple model to identify important interactions
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Get importance of interaction features
        interaction_importance = {}
        for name in interaction_names:
            idx = list(X_train.columns).index(name)
            interaction_importance[name] = model.feature_importances_[idx]
        
        # Sort and display
        sorted_interactions = sorted(
            interaction_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"\n🎯 TOP 10 FEATURE INTERACTIONS:\n")
        for i, (interaction, importance) in enumerate(sorted_interactions[:10], 1):
            feat1, feat2 = interaction.split('×')
            print(f"{i:2d}. {interaction:50s} → Importance: {importance:.4f}")
            
            # Football interpretation
            if 'speed' in interaction.lower() and 'separation' in interaction.lower():
                print(f"    💡 Fast receivers who get open = elite deep threats")
            elif 'route' in interaction.lower() and 'yac' in interaction.lower():
                print(f"    💡 Versatile route runners who make plays after catch")
            elif 'volume' in interaction.lower() and ('yac' in interaction.lower() or 'separation' in interaction.lower()):
                print(f"    💡 High-volume players maintaining production = durability")
            print()
        
        # Compare performance
        y_pred_without = self.best_model.predict(self.results['X_test'])
        y_pred_with = model.predict(X_test)
        
        r2_without = r2_score(y_test, y_pred_with)
        
        print(f"\n📈 IMPACT OF INTERACTIONS:")
        print(f"   Original R²: {self.results['model_comparison'].iloc[0]['Test R²']:.3f}")
        print(f"   With Interactions: {r2_without:.3f}")
        
        self.results['interactions'] = {
            'top_interactions': sorted_interactions[:10],
            'r2_improvement': r2_without - self.results['model_comparison'].iloc[0]['Test R²']
        }
        
        return self
    
    def shap_analysis(self, X, sample_size=100):
        """
        SHAP (SHapley Additive exPlanations) values for model interpretability.
        
        This is CUTTING EDGE interpretability:
        - Shows feature contribution for each prediction
        - Reveals non-linear effects
        - Identifies interaction effects
        - Industry standard for ML interpretability
        """
        if not SHAP_AVAILABLE:
            print("\n⚠️  SHAP not installed. Install with: pip install shap")
            return self
        
        print("\n" + "="*80)
        print("SHAP ANALYSIS: EXPLAINING INDIVIDUAL PREDICTIONS")
        print("="*80)
        
        print("\n🔬 Computing SHAP values...")
        print("   (This may take a minute for complex models)")
        
        # Sample data for efficiency
        X_sample = X.sample(min(sample_size, len(X)), random_state=42)
        
        # Create explainer
        explainer = shap.TreeExplainer(self.best_model)
        shap_values = explainer.shap_values(X_sample)
        
        # Summary statistics
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        print("\n📊 SHAP-BASED FEATURE IMPORTANCE:\n")
        print(feature_importance.head(15).to_string(index=False))
        
        print("\n💡 INTERPRETATION:")
        print("   • SHAP values show ACTUAL impact on predictions")
        print("   • Accounts for feature interactions automatically")
        print("   • Can identify positive vs negative effects")
        print("   • More accurate than standard feature importance")
        
        self.results['shap'] = {
            'values': shap_values,
            'feature_importance': feature_importance,
            'explainer': explainer,
            'X_sample': X_sample
        }
        
        print("\n✓ Use shap_values for visualization (e.g., summary plot, waterfall plot)")
        
        return self
    
    def quantify_uncertainty(self, X, y, n_iterations=100):
        """
        Bootstrap confidence intervals for predictions.
        
        Demonstrates:
        - Understanding of uncertainty quantification
        - Honesty about prediction confidence
        - Statistical rigor beyond point estimates
        """
        print("\n" + "="*80)
        print("UNCERTAINTY QUANTIFICATION VIA BOOTSTRAPPING")
        print("="*80)
        
        print(f"\n🔁 Running {n_iterations} bootstrap iterations...")
        
        from sklearn.utils import resample
        
        X_test = self.results['X_test']
        y_test = self.results['y_test']
        
        bootstrap_predictions = []
        bootstrap_r2s = []
        
        for i in range(n_iterations):
            # Resample training data
            X_boot, y_boot = resample(
                self.results['X_train'], 
                self.results['y_train'],
                random_state=i
            )
            
            # Train model on bootstrap sample
            model = type(self.best_model)(**self.best_model.get_params())
            model.fit(X_boot, y_boot)
            
            # Predict on test set
            y_pred = model.predict(X_test)
            bootstrap_predictions.append(y_pred)
            bootstrap_r2s.append(r2_score(y_test, y_pred))
        
        # Calculate confidence intervals
        bootstrap_predictions = np.array(bootstrap_predictions)
        pred_mean = bootstrap_predictions.mean(axis=0)
        pred_lower = np.percentile(bootstrap_predictions, 2.5, axis=0)
        pred_upper = np.percentile(bootstrap_predictions, 97.5, axis=0)
        
        print(f"\n📊 UNCERTAINTY METRICS:")
        print(f"   R² Mean: {np.mean(bootstrap_r2s):.3f}")
        print(f"   R² Std: {np.std(bootstrap_r2s):.3f}")
        print(f"   R² 95% CI: [{np.percentile(bootstrap_r2s, 2.5):.3f}, {np.percentile(bootstrap_r2s, 97.5):.3f}]")
        
        # Average prediction interval width
        avg_interval_width = np.mean(pred_upper - pred_lower)
        print(f"\n   Average 95% Prediction Interval Width: {avg_interval_width:.3f}")
        print(f"   (Lower = more confident predictions)")
        
        # Coverage (what % of actuals fall in intervals)
        coverage = np.mean((y_test >= pred_lower) & (y_test <= pred_upper))
        print(f"\n   Empirical Coverage: {coverage:.1%}")
        print(f"   Target Coverage: 95%")
        print(f"   {'✓ Well-calibrated!' if abs(coverage - 0.95) < 0.05 else '⚠️ May need recalibration'}")
        
        self.results['uncertainty'] = {
            'bootstrap_predictions': bootstrap_predictions,
            'pred_mean': pred_mean,
            'pred_lower': pred_lower,
            'pred_upper': pred_upper,
            'bootstrap_r2s': bootstrap_r2s,
            'coverage': coverage
        }
        
        return self


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

def run_advanced_analysis(X, y):
    """
    Complete advanced analysis pipeline.
    """
    pipeline = AdvancedModelingPipeline()
    
    # Step 1: Build and compare multiple models
    pipeline.build_model_suite()
    pipeline.compare_models(X, y)
    
    # Step 2: Hyperparameter tuning for best model
    best_model_name = pipeline.results['model_comparison'].iloc[0]['Model']
    if best_model_name in ['xgboost', 'lightgbm']:
        pipeline.hyperparameter_tuning(X, y, model_name=best_model_name)
        pipeline.compare_models(X, y)  # Re-compare with tuned model
    
    # Step 3: Build ensemble
    pipeline.build_ensemble(X, y, top_n=3)
    
    # Step 4: Feature interactions
    pipeline.analyze_feature_interactions(X, y)
    
    # Step 5: SHAP analysis (if available)
    if SHAP_AVAILABLE:
        pipeline.shap_analysis(X)
    
    # Step 6: Uncertainty quantification
    pipeline.quantify_uncertainty(X, y, n_iterations=50)
    
    return pipeline


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║              ADVANCED MULTI-MODEL ANALYSIS PIPELINE                      ║
    ║                                                                          ║
    ║     Demonstrating Technical Depth + Interpretability                     ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    
    This pipeline includes:
    
    ✓ Multiple model comparison (6 algorithms)
    ✓ Hyperparameter optimization
    ✓ Ensemble learning
    ✓ Feature interaction analysis
    ✓ SHAP interpretability
    ✓ Uncertainty quantification
    
    Usage:
        from advanced_modeling import run_advanced_analysis
        
        pipeline = run_advanced_analysis(X, y)
        results = pipeline.results
    """)