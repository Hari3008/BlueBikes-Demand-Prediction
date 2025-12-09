import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Simple logging helper
# ---------------------------------------------------------------------------
def log(msg):
    """Print log message with timestamp and force flush."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================
log("Loading data...")
df = pd.read_csv('bluebikes_ml_ready.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by timestamp to maintain temporal order
df = df.sort_values('timestamp').reset_index(drop=True)

log(f"Dataset shape: {df.shape}")
log(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Separate features and target
target_col = 'demand'

# Identify and exclude non-numeric columns
non_numeric_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
log(f"Non-numeric columns found: {non_numeric_cols}")

# Exclude timestamp, target, station_id, and all non-numeric columns
exclude_cols = ['timestamp', target_col, 'station_id'] + non_numeric_cols
feature_cols = [col for col in df.columns if col not in exclude_cols]

log(f"Number of numeric features: {len(feature_cols)}")
log(f"Features being used (first 10): {feature_cols[:10]}")

# Select only numeric features
X = df[feature_cols].copy()
y = df[target_col].copy()

# Check for any remaining non-numeric data
log("Data types in X:")
print(X.dtypes.value_counts())

# Convert any remaining non-numeric to numeric (safety check)
for col in X.columns:
    if X[col].dtype == 'object':
        log(f"WARNING: Column '{col}' is still object type, attempting conversion...")
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Check for NaN values
nan_counts = X.isna().sum()
if nan_counts.sum() > 0:
    log(f"NaN values found in {(nan_counts > 0).sum()} columns. Filling NaN values with 0...")
    X = X.fillna(0)
else:
    log("No NaN values found in X.")

# >>> MEMORY OPTIMIZATION: use float32 instead of float64 <<<
log("Casting feature matrix to float32 to reduce memory usage...")
X = X.astype('float32')

# Time-series split (70% train, 15% validation, 15% test)
train_size = int(0.70 * len(df))
val_size = int(0.85 * len(df))

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:val_size]
y_val = y[train_size:val_size]

X_test = X[val_size:]
y_test = y[val_size:]

log(f"Train size: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
log(f"Validation size: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%)")
log(f"Test size: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

# ============================================================================
# BASELINE MODEL (No Hyperparameter Tuning)
# ============================================================================
print("\n" + "="*80)
print("BASELINE MODEL - Linear Regression with Polynomial Features (Degree 2)")
print("="*80)

baseline_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=1.0, random_state=42))
])

log("Starting baseline model training...")
baseline_pipeline.fit(X_train, y_train)
log("Baseline model training completed.")

# Evaluate baseline model
log("Making baseline predictions on train / val / test...")
y_train_pred = baseline_pipeline.predict(X_train)
y_val_pred = baseline_pipeline.predict(X_val)
y_test_pred = baseline_pipeline.predict(X_test)
log("Baseline predictions completed.")

baseline_metrics = {
    'train': {
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred),
        'r2': r2_score(y_train, y_train_pred)
    },
    'validation': {
        'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'mae': mean_absolute_error(y_val, y_val_pred),
        'r2': r2_score(y_val, y_val_pred)
    },
    'test': {
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae': mean_absolute_error(y_test, y_test_pred),
        'r2': r2_score(y_test, y_test_pred)
    }
}

print("\nBaseline Model Performance:")
print("-" * 80)
for split, metrics in baseline_metrics.items():
    print(f"\n{split.upper()} SET:")
    print(f"  RMSE: {metrics['rmse']:.3f}")
    print(f"  MAE:  {metrics['mae']:.3f}")
    print(f"  R²:   {metrics['r2']:.3f}")

# Save baseline model
log("Saving baseline model to 'baseline_lr_poly_model.pkl'...")
with open('baseline_lr_poly_model.pkl', 'wb') as f:
    pickle.dump(baseline_pipeline, f)
log("Baseline model saved.")

# ============================================================================
# OPTIMIZED MODEL (With Cross-Validation Hyperparameter Tuning)
# ============================================================================
print("\n" + "="*80)
print("OPTIMIZED MODEL - Hyperparameter Tuning with Time Series Cross-Validation")
print("="*80)

# >>> LIGHTER GRID FOR LAPTOP-FRIENDLY RUN <<<
param_grid = {
    'poly__degree': [1, 2],                 # no degree 3
    'ridge__alpha': [0.01, 0.1, 1.0, 10.0]  # fewer alphas
}

print("\nParameter Grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

log("Creating tuning pipeline for GridSearchCV...")
tuning_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(include_bias=False)),
    ('ridge', Ridge(random_state=42))
])

# Use TimeSeriesSplit for cross-validation to respect temporal order
tscv = TimeSeriesSplit(n_splits=3)  # lighter than 5 splits
log(f"TimeSeriesSplit created with n_splits={tscv.n_splits}.")

# Combine train and validation for CV
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
log(f"Train+Val combined for CV: X_train_val shape={X_train_val.shape}, y_train_val length={len(y_train_val)}")

print(f"\nPerforming GridSearchCV with {tscv.n_splits}-fold Time Series CV...")
print("Running fits sequentially to reduce memory usage...")

log("Initializing GridSearchCV object...")
grid_search = GridSearchCV(
    tuning_pipeline,
    param_grid,
    cv=tscv,
    scoring='neg_root_mean_squared_error',
    n_jobs=1,        # run one job at a time (no parallelism)
    verbose=3        # detailed progress logs
)

log("Starting GridSearchCV fit on train+val data...")
grid_search.fit(X_train_val, y_train_val)
log("GridSearchCV fit completed.")

print("\n✓ Hyperparameter tuning complete!")
print("\nBest Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV RMSE: {-grid_search.best_score_:.3f}")

# Get the best model
log("Extracting best estimator from GridSearchCV...")
optimized_pipeline = grid_search.best_estimator_

# Evaluate optimized model
log("Making predictions with optimized model on train+val and test...")
y_train_val_pred = optimized_pipeline.predict(X_train_val)
y_test_pred_opt = optimized_pipeline.predict(X_test)
log("Optimized model predictions completed.")

optimized_metrics = {
    'train_val': {
        'rmse': np.sqrt(mean_squared_error(y_train_val, y_train_val_pred)),
        'mae': mean_absolute_error(y_train_val, y_train_val_pred),
        'r2': r2_score(y_train_val, y_train_val_pred)
    },
    'test': {
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred_opt)),
        'mae': mean_absolute_error(y_test, y_test_pred_opt),
        'r2': r2_score(y_test, y_test_pred_opt)
    }
}

print("\nOptimized Model Performance:")
print("-" * 80)
for split, metrics in optimized_metrics.items():
    print(f"\n{split.upper().replace('_', '+')} SET:")
    print(f"  RMSE: {metrics['rmse']:.3f}")
    print(f"  MAE:  {metrics['mae']:.3f}")
    print(f"  R²:   {metrics['r2']:.3f}")

# Save optimized model
log("Saving optimized model to 'optimized_lr_poly_model.pkl'...")
with open('optimized_lr_poly_model.pkl', 'wb') as f:
    pickle.dump(optimized_pipeline, f)
log("Optimized model saved.")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison_df = pd.DataFrame({
    'Metric': ['Test RMSE:', 'Test MAE:', 'Test R2:'],
    'Baseline': [
        f"{baseline_metrics['test']['rmse']:.3f}",
        f"{baseline_metrics['test']['mae']:.3f}",
        f"{baseline_metrics['test']['r2']:.3f}"
    ],
    'Optimized': [
        f"{optimized_metrics['test']['rmse']:.3f}",
        f"{optimized_metrics['test']['mae']:.3f}",
        f"{optimized_metrics['test']['r2']:.3f}"
    ],
    'Improvement': [
        f"{(baseline_metrics['test']['rmse'] - optimized_metrics['test']['rmse']) / baseline_metrics['test']['rmse'] * 100:+.1f}%",
        f"{(baseline_metrics['test']['mae'] - optimized_metrics['test']['mae']) / baseline_metrics['test']['mae'] * 100:+.1f}%",
        f"{(optimized_metrics['test']['r2'] - baseline_metrics['test']['r2']) / baseline_metrics['test']['r2'] * 100:+.1f}%"
    ]
})

log("Model comparison table:")
print("\n" + comparison_df.to_string(index=False))

# ============================================================================
# CV RESULTS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)

log("Processing cv_results_ from GridSearchCV...")
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results['mean_rmse'] = -cv_results['mean_test_score']
cv_results['std_rmse'] = cv_results['std_test_score']

# Get top 10 configurations (or fewer if less than 10)
top_configs = cv_results.nsmallest(10, 'mean_rmse')[
    ['param_poly__degree', 'param_ridge__alpha', 'mean_rmse', 'std_rmse', 'rank_test_score']
].copy()

log("Top configurations from CV:")
print("\nTop Configurations:")
print(top_configs.to_string(index=False))

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("TOP 15 FEATURE IMPORTANCES")
print("="*80)

log("Computing feature importances from optimized model...")
poly_features = optimized_pipeline.named_steps['poly']
ridge_model = optimized_pipeline.named_steps['ridge']

feature_names = poly_features.get_feature_names_out(feature_cols)
coefficients = ridge_model.coef_

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
}).sort_values('abs_coefficient', ascending=False)

print()
for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['feature']:<50} : {row['abs_coefficient']:.4f}")

log("Saving feature importance to 'feature_importance.csv'...")
feature_importance.to_csv('feature_importance.csv', index=False)
log("Feature importance saved.")

# ============================================================================
# SAVE COMPREHENSIVE METRICS LOG
# ============================================================================
log("Building comprehensive metrics log dictionary...")
metrics_log = {
    'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'num_features': len(feature_cols),
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}"
    },
    'baseline_model': {
        'description': 'Linear Regression with Degree-2 Polynomial Features',
        'parameters': {
            'poly_degree': 2,
            'ridge_alpha': 1.0
        },
        'metrics': baseline_metrics
    },
    'optimized_model': {
        'description': 'Hyperparameter-tuned with GridSearchCV and TimeSeriesSplit',
        'best_parameters': grid_search.best_params_,
        'cv_best_score': float(-grid_search.best_score_),
        'metrics': optimized_metrics
    },
    'cv_details': {
        'cv_strategy': 'TimeSeriesSplit',
        'n_splits': tscv.n_splits,
        'param_grid': param_grid,
        'total_fits': len(cv_results)
    },
    'comparison': {
        'test_rmse_improvement': f"{(baseline_metrics['test']['rmse'] - optimized_metrics['test']['rmse']) / baseline_metrics['test']['rmse'] * 100:.2f}%",
        'test_mae_improvement': f"{(baseline_metrics['test']['mae'] - optimized_metrics['test']['mae']) / baseline_metrics['test']['mae'] * 100:.2f}%",
        'test_r2_improvement': f"{(optimized_metrics['test']['r2'] - baseline_metrics['test']['r2']) / baseline_metrics['test']['r2'] * 100:.2f}%"
    }
}

log("Saving metrics log to 'model_metrics_log.json'...")
with open('model_metrics_log.json', 'w') as f:
    json.dump(metrics_log, f, indent=4)
log("Metrics log saved.")

print("\n" + "="*80)
print("ALL FILES SAVED SUCCESSFULLY!")
print("="*80)
print("\nGenerated Files:")
print("  1. baseline_lr_poly_model.pkl - Baseline model")
print("  2. optimized_lr_poly_model.pkl - Hyperparameter-tuned model")
print("  3. model_metrics_log.json - Comprehensive metrics and experiment details")
print("  4. feature_importance.csv - Feature importance ranking")
print("\nYou can now use these models for your dashboard!")
