import sys
from pathlib import Path

# Determine project root dynamically (two levels up from analyses/)
project_root = Path(__file__).resolve().parents[1]  # adjust if depth changes
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.base import clone
from category_encoders.target_encoder import TargetEncoder

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import RandomizedSearchCV

from d100project.preprocessing._log_transformer import LogTransformer
from d100project.preprocessing._one_hot_encode import ListOneHotEncoder
from d100project.preprocessing._month_to_season import MonthToSeasonTransformer

from d100project.data._create_sample_split import create_sample_split
from d100project.data._load_cleaned_parquet import load_cleaned_parquet

# Load cleaned data
df = load_cleaned_parquet()

# Create train/test split column
df = create_sample_split(df, id_column="id", training_frac=0.8)

# create a column for ln(revenue)
df['ln_revenue'] = np.log1p(df['revenue'])

# training and testing dataframes
df_train = df[df["sample"] == "train"].copy()
df_test = df[df["sample"] == "test"].copy()

target = "revenue"

numeric_features = ["budget", "runtime", "year"]
list_features = ["genres_list"]  # OHE only for genres
high_cardinality = ["original_language"]  # target encoding
count_features = ["n_production_companies", "n_production_countries", "n_spoken_languages"]
month_feature = "month"


all_features = numeric_features + list_features + [month_feature] + high_cardinality + count_features

X_train = df_train[all_features]
y_train = df_train['ln_revenue']

X_test = df_test[all_features]
y_test = df_test['ln_revenue']


# Preprocessing pipelines for different feature types
numeric_transformer = Pipeline(steps=[
    ('log_transform', LogTransformer()),
    ('scaler', StandardScaler())
])
list_transformer = Pipeline(steps=[
    ('one_hot_encode', ListOneHotEncoder(columns=list_features))
])
month_transformer = Pipeline(steps=[
    ('month_to_season', MonthToSeasonTransformer(month_column=month_feature))
])

# Preprocessors

preprocessor = ColumnTransformer(
    transformers=[
        ('log_numeric', LogTransformer(), ['budget']),
        ('num', StandardScaler(), ['runtime','year']),
        ('list', ListOneHotEncoder(columns=list_features), list_features),
        ('month', MonthToSeasonTransformer(month_column=month_feature), [month_feature])
    ]
)

preprocessor_lgbm = ColumnTransformer(
    transformers=[
        ("log_budget", Pipeline([("log", LogTransformer())]), ["budget"]),
        ("num", "passthrough", ["runtime", "year"] + count_features),
        ("te", TargetEncoder(smoothing=10, min_samples_leaf=20), ["original_language"]),
        ("genres", ListOneHotEncoder(columns=list_features), ["genres_list"]),
        ("month", MonthToSeasonTransformer(month_column="month"), [month_feature])
    ]
)


# FIT PREPROCESSORS ONCE & TRANSFORM DATA BEFORE GRID SEARCH

preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

preprocessor_lgbm.fit(X_train, y_train)
X_train_lgbm = preprocessor_lgbm.transform(X_train)
X_test_lgbm = preprocessor_lgbm.transform(X_test)

# Create the model pipelines

# GLM
GLM_pipeline = ElasticNet(max_iter=10000)

# Hyperparameter grid for GLM
glm_param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.0, 0.5, 1.0]  # 0=Ridge, 1=Lasso
}


# Hyperparameter grid for LightGBM
lgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'num_leaves': [31, 50],
    'min_child_weight': [1, 5]
}

# GLM hyperparameter tuning
glm_search = GridSearchCV(
    estimator=GLM_pipeline,
    param_grid=glm_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# LightGBM
LGBM_pipeline = lgb.LGBMRegressor(objective='regression', n_estimators=5000)


lgb_search = RandomizedSearchCV(
    estimator=LGBM_pipeline,
    param_distributions=lgb_param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_iter=10,
    verbose=2,
    n_jobs=-1
)


# Fit GLM with GridSearchCV
glm_search.fit(X_train_transformed, y_train)
print("Best GLM params:", glm_search.best_params_)

# Fit LGBM with GridSearchCV
lgb_search.fit(
    X_train_lgbm,
    y_train,
    eval_set=[(X_test_lgbm, y_test)],
    eval_metric="mse",
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(100)
    ]
)
print("Best LGBM params:", lgb_search.best_params_)

print("Training complete.")

# ----------------------------
# Predictions
# ----------------------------
y_pred_GLM = glm_search.best_estimator_.predict(X_test_transformed)
y_pred_LGBM = lgb_search.best_estimator_.predict(X_test_lgbm)

# ----------------------------
# Evaluate MSE and R^2
# ----------------------------
mse_glm = mean_squared_error(y_test, y_pred_GLM)
r2_glm = r2_score(y_test, y_pred_GLM)


mse_lgbm = mean_squared_error(y_test, y_pred_LGBM)
r2_lgbm = r2_score(y_test, y_pred_LGBM)

print(f"GLM - MSE: {mse_glm:.2f}, R^2: {r2_glm:.3f}")
print(f"LGBM - MSE: {mse_lgbm:.2f}, R^2: {r2_lgbm:.3f}")

# Save or replace predictions in test DataFrame
if 'pred_glm' in df_test.columns:
    df_test['pred_glm'] = y_pred_GLM  # replace existing
else:
    df_test['pred_glm'] = y_pred_GLM  # create new column

if 'pred_lgbm' in df_test.columns:
    df_test['pred_lgbm'] = y_pred_LGBM
else:
    df_test['pred_lgbm'] = y_pred_LGBM


# Save the test DataFrame with predictions
project_root = Path(__file__).resolve().parents[1]  # adjust if needed
output_path = project_root / "d100project" / "evaluation" / "df_test_with_predictions.parquet"
df_test.to_parquet(output_path, index=False)
print(f"Saved test set with predictions to {output_path}")