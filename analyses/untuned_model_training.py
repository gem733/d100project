import sys
from pathlib import Path

# Set up project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


import lightgbm as lgb
import numpy as np
import shap
from category_encoders.target_encoder import TargetEncoder
from joblib import dump
from lightgbm import early_stopping, log_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from d100project.data._create_sample_split import create_sample_split
from d100project.data._load_cleaned_parquet import load_cleaned_parquet
from d100project.preprocessing._log_transformer import LogTransformer
from d100project.preprocessing._month_to_season import MonthToSeasonTransformer
from d100project.preprocessing._one_hot_encode import ListOneHotEncoder

# Load data
df = load_cleaned_parquet()

# Train/test split indicator
df = create_sample_split(df, id_column="id", training_frac=0.8)

# Log target (for evaluation)
df["ln_revenue"] = np.log1p(df["revenue"])

df_train = df[df["sample"] == "train"].copy()
df_test = df[df["sample"] == "test"].copy()

# Features & target
target = "revenue"

numeric_features = ["budget", "runtime", "year"]
list_features = ["genres_list"]
high_cardinality = ["original_language"]
count_features = [
    "n_production_companies",
    "n_production_countries",
    "n_spoken_languages",
]
month_feature = "month"
indicators = ['runtime_was_missing', 'budget_was_missing']

all_features = (
    numeric_features
    + list_features
    + [month_feature]
    + high_cardinality
    + count_features
    + indicators
)

X_train = df_train[all_features]
y_train = df_train["ln_revenue"]

X_test = df_test[all_features]
y_test = df_test["ln_revenue"]

# Preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ("log_budget", LogTransformer(), ["budget"]),
        ("num", StandardScaler(), ["runtime", "year"]),
        ("genres", ListOneHotEncoder(columns=list_features), list_features),
        ("month", MonthToSeasonTransformer(month_column=month_feature), [month_feature]),
        ("indicators", "passthrough", indicators)
    ]
)

preprocessor_lgbm = ColumnTransformer(
    transformers=[
        ("log_budget", LogTransformer(), ["budget"]),
        ("num", "passthrough", ["runtime", "year"] + count_features),
        ("te", TargetEncoder(smoothing=10, min_samples_leaf=20), high_cardinality),
        ("genres", ListOneHotEncoder(columns=list_features), list_features),
        ("month", MonthToSeasonTransformer(month_column=month_feature), [month_feature]),
        ("indicators", "passthrough", indicators)
    ]
)

# Fit preprocessors
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

X_train_lgbm = preprocessor_lgbm.fit_transform(X_train, y_train)
X_test_lgbm = preprocessor_lgbm.transform(X_test)

# Base models (NO hyperparameter tuning)
glm_model = ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,
    max_iter=10000,
    random_state=42,
)

lgbm_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=5000,
    learning_rate=0.05,
    random_state=42,
)

# Fit models
glm_model.fit(X_train_transformed, y_train)

lgbm_model.fit(
    X_train_lgbm,
    y_train,
    eval_set=[(X_test_lgbm, y_test)],
    eval_metric="mse",
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(100),
    ],
)

# Predictions
y_pred_glm = glm_model.predict(X_test_transformed)
y_pred_lgbm = lgbm_model.predict(X_test_lgbm)

# Evaluation
mse_glm = mean_squared_error(y_test, y_pred_glm)
r2_glm = r2_score(y_test, y_pred_glm)

mse_lgbm = mean_squared_error(y_test, y_pred_lgbm)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print(f"GLM  - MSE: {mse_glm:.2f}, R²: {r2_glm:.3f}")
print(f"LGBM - MSE: {mse_lgbm:.2f}, R²: {r2_lgbm:.3f}")
