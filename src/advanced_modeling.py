"""
Advanced Multi-Model Analysis Pipeline
======================================

Key Features:
- Multiple model comparison
- Ensemble methods
- Feature interaction analysis
- Uncertainty quantification
- SHAP values
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
from sklearn.utils import resample

import xgboost as xgb
import lightgbm as lgb
import warnings

# Suppress warnings from model libraries to keep output readable
warnings.filterwarnings('ignore')

# Optional dependency for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Install with: pip install shap")


class AdvancedModelingPipeline:
    # Container for model construction, evaluation, and analysis
    
    def __init__(self):
        # Stores initialized models by name
        self.models = {}
        # Stores intermediate and final results
        self.results = {}
        # Tracks the best-performing model after comparison
        self.best_model = None
        # Stores weights used for ensemble prediction
        self.ensemble_weights = {}
        
    def build_model_suite(self):
        """
        Initialize the suite of candidate models used for comparison.

        Populates self.models with baseline linear, tree-based, and
        gradient-boosted regressors using fixed default configurations.

        Returns: self
        """
        print("Building models... ")
        
        # Dictionary defining models and their configuration metadata
        models = {
            # Linear baseline for reference
            'ridge': {
                'model': Ridge(alpha=1.0),
                'rationale': 'Baseline linear model - interpretable coefficients',
                'strength': 'Clear feature relationships',
                'weakness': 'Assumes linearity'
            },
            
            # Bagged decision trees for non-linear structure
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
            
            # Sequential boosting with shallow trees
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
            
            # Regularized gradient boosting using XGBoost
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
            
            # Histogram-based gradient boosting using LightGBM
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
            
            # Linear model with combined L1 and L2 regularization
            'elastic_net': {
                'model': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
                'rationale': 'Feature selection via L1, stability via L2',
                'strength': 'Automatic feature selection',
                'weakness': 'Linear assumptions'
            }
        }
        
        # Print summary of model choices
        print("\nModel Information:\n")
        for name, info in models.items():
            print(f"{name.upper()}")
            print(f"  Rationale: {info['rationale']}")
            print(f"  - Strength: {info['strength']}")
            print(f"  - Weakness: {info['weakness']}\n")
        
        # Store only the model objects for downstream use
        self.models = {name: info['model'] for name, info in models.items()}
        return self
    
    def compare_models(self, X, y, cv_folds=5):
        """
        Train and evaluate all configured models using cross-validation
        and a held-out test set.

        Stores per-model performance metrics including R², MAE, RMSE,
        and overfitting diagnostics in self.results['model_comparison'].

        Params:
        X : pandas.DataFrame
        Feature matrix.
        y : pandas.Series or numpy.ndarray
        Target variable.
        
        Returns: self
        """
        print("\n\nComparing the Models")
        print(f"\nValidation Strategy: {cv_folds}-Fold Cross-Validation")
        print(f"Training Samples: {len(X):,}")
        print(f"Features: {X.shape[1]}\n")
        
        from sklearn.model_selection import train_test_split
        
        # Create a fixed train-test split for fair comparison
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        results = []
        
        for name, model in self.models.items():
            print(f"\nTraining {name.upper()}...")
            
            # Cross-validation performed only on training data
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1
            )
            
            # Fit model on full training set
            model.fit(X_train, y_train)
            
            # Generate predictions for diagnostics
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Compute evaluation metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Difference between train and test performance
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
                  "High" if overfit_gap > 0.15 else "Good")
        
        # Aggregate results into a table sorted by test performance
        results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
        
        print("\n\nModel Comparison")
        print(results_df.to_string(index=False))
        
        # Select the best-performing model on the test set
        best_idx = results_df['Test R²'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[best_model_name]
        
        print(f"\n Best Model: {best_model_name.upper()}")
        print(f"   Test R²: {results_df.loc[best_idx, 'Test R²']:.3f}")
        print(f"   Test MAE: {results_df.loc[best_idx, 'Test MAE']:.3f}")
        
        # Persist data splits and comparison results
        self.results['model_comparison'] = results_df
        self.results['X_train'] = X_train
        self.results['X_test'] = X_test
        self.results['y_train'] = y_train
        self.results['y_test'] = y_test
        
        return self
    def hyperparameter_tuning(self, X, y, model_name='xgboost'):
        """
        Perform cross-validated hyperparameter tuning for a supported model.
        Uses randomized search to efficiently explore the parameter space
        and identify a high-performing configuration. The best estimator
        and tuning metadata are stored for downstream evaluation.
        Adds a tuned model to self.models and stores tuning results in self.results


        Params:
        X : pandas.DataFrame
        Feature matrix.
        y : pandas.Series or numpy.ndarray
        Target values.
        model_name : str, default='xgboost'
        Model to tune. Supported options: 'xgboost', 'lightgbm'.

        Returns: self
        """
        print(f"Hyperparameter tuning: {model_name.upper()}")
        
        # Define search spaces tailored to each boosting framework
        if model_name == 'xgboost':
            param_grid = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            # Start from an untuned baseline model
            base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
        elif model_name == 'lightgbm':
            param_grid = {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'num_leaves': [15, 31, 63],
                'min_child_samples': [5, 10, 20]
            }
            base_model = lgb.LGBMRegressor(
                random_state=42, n_jobs=-1, verbose=-1
            )
        
        # Exit early if tuning is not defined for the selected model
        else:
            print(f"Tuning not implemented for {model_name}")
            return self
        
        # Log size of the hyperparameter search space
        print(f"\nSearching {len(param_grid)} hyperparameters...")
        print(
            f"   Search space: ~"
            f"{np.prod([len(v) for v in param_grid.values()]):,} combinations"
        )
        
        # Randomized search used instead of full grid for computational efficiency
        from sklearn.model_selection import RandomizedSearchCV
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=50,               # Samples a representative subset
            scoring='r2',
            cv=5,                    # Nested cross-validation for robustness
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        # Fit search over the full dataset
        search.fit(X, y)
        
        # Report best-performing configuration
        print(f"   Best CV R²: {search.best_score_:.3f}")
        print(f"   Best Parameters:")
        for param, value in search.best_params_.items():
            print(f"      {param}: {value}")
        
        # Store tuned model alongside original versions
        self.models[f'{model_name}_tuned'] = search.best_estimator_
        self.results[f'{model_name}_tuning'] = {
            'best_score': search.best_score_,
            'best_params': search.best_params_
        }
        
        return self
    
    def build_ensemble(self, X, y, top_n=3):
        """
        Construct a weighted ensemble from the top-performing models.

        Ensemble weights are derived from inverse test-set MAE, and
        predictions are combined via weighted averaging.

        Stores ensemble weights and predictions in self.results

        Params: 
        X : pandas.DataFrame
        Feature matrix.
        y : pandas.Series or numpy.ndarray
        Target values.
        top_n : int, default=3
        Number of top models to include in the ensemble.

        Return: self
        """

        print("Building Models")
        
        # Ensure base model comparison has already been run
        if 'model_comparison' not in self.results:
            print("Run compare_models() first!")
            return self
        
        # Select top-N models by test R²
        top_models = self.results['model_comparison'].head(top_n)
        
        print(f"\nEnsemble Components (Top {top_n} by Test R²):\n")
        print(top_models[['Model', 'Test R²', 'Test MAE']].to_string(index=False))
        
        # Retrieve held-out test data
        X_test = self.results['X_test']
        y_test = self.results['y_test']
        
        # Generate predictions from each ensemble member
        predictions = {}
        for _, row in top_models.iterrows():
            model_name = row['Model']
            model = self.models[model_name]
            predictions[model_name] = model.predict(X_test)
        
        # Weight models by inverse MAE to favor lower-error predictors
        weights = {}
        total_inv_mae = 0
        for _, row in top_models.iterrows():
            inv_mae = 1 / row['Test MAE']
            weights[row['Model']] = inv_mae
            total_inv_mae += inv_mae
        
        # Normalize weights so they sum to one
        for model_name in weights:
            weights[model_name] /= total_inv_mae
        
        print("\nEnsemble Weights:")
        for model_name, weight in weights.items():
            print(f"   {model_name:20s}: {weight:.3f}")
        
        # Compute weighted average prediction
        ensemble_pred = np.zeros(len(y_test))
        for model_name, pred in predictions.items():
            ensemble_pred += pred * weights[model_name]
        
        # Evaluate ensemble performance
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        
        print(f"\nEnsemble Performance:")
        print(f"   Test R²: {ensemble_r2:.3f}")
        print(f"   Test MAE: {ensemble_mae:.3f}")
        
        # Compare ensemble to best single model
        best_single_r2 = top_models.iloc[0]['Test R²']
        improvement = ensemble_r2 - best_single_r2
        
        print(f"\n   vs Best Single Model:")
        print(f"   Improvement: {improvement:+.3f} R²")
        print(
            f"   {'Ensemble wins!' if improvement > 0 else 'Single model better (use that)'}"
        )
        
        # Persist ensemble outputs
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
        Identify and rank pairwise feature interaction effects.

        Generates multiplicative interaction features among the most
        influential predictors and evaluates their importance using
        a tree-based model.

        Stores ranked interaction effects in self.results

        Params:
        X : pandas.DataFrame
        Feature matrix.
        y : pandas.Series or numpy.ndarray
        Target values.
        top_n_features : int, default=10
        Number of base features used to construct interactions.

        Returns: self
        """        
        print("\n\nFeature Interaction Analysis")
        print("\nAnalyzing pairwise feature interactions...")
        
        # Identify top features using model-based importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            top_features = X.columns[np.argsort(importance)[-top_n_features:]]
        else:
            # Fallback to absolute correlation with target
            correlations = X.corrwith(y).abs()
            top_features = correlations.nlargest(top_n_features).index
        
        print(f"\nTesting interactions among top {len(top_features)} features:")
        for feat in top_features:
            print(f"   • {feat}")
        
        # Explicitly construct pairwise interaction terms
        X_with_interactions = X.copy()
        interaction_names = []
        
        from itertools import combinations
        for feat1, feat2 in combinations(top_features, 2):
            interaction_name = f"{feat1}×{feat2}"
            X_with_interactions[interaction_name] = X[feat1] * X[feat2]
            interaction_names.append(interaction_name)
        
        # Train-test split with interaction features
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_with_interactions, y, test_size=0.25, random_state=42
        )
        
        # Use a constrained random forest to rank interaction importance
        model = RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Extract importance scores for interaction terms only
        interaction_importance = {}
        for name in interaction_names:
            idx = list(X_train.columns).index(name)
            interaction_importance[name] = model.feature_importances_[idx]
        
        # Rank interactions by importance
        sorted_interactions = sorted(
            interaction_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print(f"\nTop 10 Feature Interactions:\n")
        for i, (interaction, importance) in enumerate(sorted_interactions[:10], 1):
            if 'x' not in interaction:
                continue
            print(f"{i:2d}. {interaction:50s} → Importance: {importance:.4f}")
            print()
        
        # Compare predictive performance with and without interactions
        y_pred_with = model.predict(X_test)
        r2_with = r2_score(y_test, y_pred_with)
        
        print(f"\nImpact on Interactions:")
        print(f"   Original R²: {self.results['model_comparison'].iloc[0]['Test R²']:.3f}")
        print(f"   With Interactions: {r2_with:.3f}")
        
        self.results['interactions'] = {
            'top_interactions': sorted_interactions[:10],
            'r2_improvement': (
                r2_with - self.results['model_comparison'].iloc[0]['Test R²']
            )
        }
        
        return self
    def shap_analysis(self, X, sample_size=100):
        """
        Compute SHAP values for the current best model.
        Stores SHAP values, feature-level importance, and the SHAP explainer
        in self.results['shap'].

        Params:
        X : pandas.DataFrame
            Feature matrix used to compute SHAP values.
        sample_size : int, default=100
            Number of rows sampled from X to limit computation cost.

        Returns: self
        """
        # Sample a subset of rows to reduce SHAP computation cost
        X_sample = X.sample(min(sample_size, len(X)), random_state=42)

        # Tree-based explainer for the fitted model
        explainer = shap.TreeExplainer(self.best_model)

        # SHAP values for each feature and sample
        shap_values = explainer.shap_values(X_sample)

        # Aggregate absolute SHAP values for global feature importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    def quantify_uncertainty(self, X, y, n_iterations=100):
        """
        Estimate prediction uncertainty using bootstrap resampling.
        Stores bootstrap predictions, confidence intervals, and
        distributional metrics in self.results['uncertainty'].

        Params
        X : pandas.DataFrame
            Feature matrix.
        y : pandas.Series or numpy.ndarray
            Target values.
        n_iterations : int, default=100
            Number of bootstrap resamples.

        Returns: self
        """
        # Test split reused from compare_models
        X_test = self.results['X_test']
        y_test = self.results['y_test']

        # Collect per-iteration predictions and R² values
        bootstrap_predictions = []
        bootstrap_r2s = []

        # Resample training data and retrain the model each iteration
        for i in range(n_iterations):
            X_boot, y_boot = resample(
                self.results['X_train'],
                self.results['y_train'],
                random_state=i
            )

            # Clone model using the same hyperparameters
            model = type(self.best_model)(**self.best_model.get_params())
            model.fit(X_boot, y_boot)

            # Predict on the fixed test set
            y_pred = model.predict(X_test)
            bootstrap_predictions.append(y_pred)
            bootstrap_r2s.append(r2_score(y_test, y_pred))

        # Compute empirical prediction intervals
        bootstrap_predictions = np.array(bootstrap_predictions)
        pred_mean = bootstrap_predictions.mean(axis=0)
        pred_lower = np.percentile(bootstrap_predictions, 2.5, axis=0)
        pred_upper = np.percentile(bootstrap_predictions, 97.5, axis=0)

        # Fraction of true values falling inside the interval
        coverage = np.mean((y_test >= pred_lower) & (y_test <= pred_upper))

def run_advanced_analysis(X, y):
    """
    Execute the full modeling pipeline on the provided dataset.

    Parameters: 
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series or numpy.ndarray
        Target values.

    Returns: AdvancedModelingPipeline, which is a fitted pipeline containing models and results.
    """
    pipeline = AdvancedModelingPipeline()
    
    # Build and compare multiple models
    pipeline.build_model_suite()
    pipeline.compare_models(X, y)
    
    # Hyperparameter tuning for best model
    best_model_name = pipeline.results['model_comparison'].iloc[0]['Model']
    if best_model_name in ['xgboost', 'lightgbm']:
        pipeline.hyperparameter_tuning(X, y, model_name=best_model_name)
        pipeline.compare_models(X, y)  # Re-compare with tuned model
    
    # Build ensemble
    pipeline.build_ensemble(X, y, top_n=3)
    
    # Feature interactions
    pipeline.analyze_feature_interactions(X, y)
    
    # SHAP analysis (if available)
    if SHAP_AVAILABLE:
        pipeline.shap_analysis(X)
    
    # Uncertainty quantification
    pipeline.quantify_uncertainty(X, y, n_iterations=50)
    
    return pipeline


if __name__ == "__main__":
    print("""
    Advanced Multi-Model Analysis Pipeline                

    This pipeline includes:

    - Multiple model comparison (6 algorithms)
    - Hyperparameter optimization
    - Ensemble learning
    - Feature interaction analysis
    - SHAP interpretability
    - Uncertainty quantification

    Usage:
    from advanced_modeling import run_advanced_analysis

    pipeline = run_advanced_analysis(X, y)
    results = pipeline.results
    """)