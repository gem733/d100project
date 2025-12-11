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

# training and testing dataframes
df_train = df[df["sample"] == "train"].copy()
df_test = df[df["sample"] == "test"].copy()

target = "revenue"

numeric_features = ["budget", "runtime", "year"]
log_transform_features = ["budget", "revenue"]
list_features = ["original_language", "genres_list", "production_companies_list", "production_countries_list", "spoken_languages_list"]
month_feature = "month"

all_features = numeric_features + list_features + [month_feature]

X_train = df_train[all_features]
y_train = np.log1p(df_train['revenue'])

X_test = df_test[all_features]
y_test  = np.log1p(df_test['revenue'])

# the number of dummy colums created by ListOneHotEncoder could lead to large overfitting for the linear model, I'll create another set of features without them.

some_features = numeric_features + [month_feature]

X_train_limited = df_train[some_features]
X_test_limited = df_test[some_features]


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
preprocessor = ColumnTransformer(
    transformers=[
        ('log_numeric', LogTransformer(), ['budget']),
        ('num', StandardScaler(), ['runtime','year']),
        ('list', ListOneHotEncoder(columns=list_features), list_features),
        ('month', MonthToSeasonTransformer(month_column=month_feature), [month_feature])
    ]
)

preprocessor_limited = ColumnTransformer(
    transformers=[
        ('log_numeric', LogTransformer(), ['budget']),
        ('num', StandardScaler(), ['runtime','year']),
        ('month', MonthToSeasonTransformer(month_column=month_feature), [month_feature])
    ]
)

# Create the model pipelines

# GLM
GLM_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_limited), # using limited preprocessor to speed up training
    ('regressor', ElasticNet(max_iter=10000))
])

GLM_limited_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_limited),
    ('regressor', ElasticNet(max_iter=10000))
])

# Hyperparameter grid for GLM
glm_param_grid = {
    'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
    'regressor__l1_ratio': [0.0, 0.5, 1.0]  # 0=Ridge, 1=Lasso
}

# Hyperparameter grid for GLM with a limited feature set
glm_limited_param_grid = {
    'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
    'regressor__l1_ratio': [0.0, 0.5, 1.0]  # 0=Ridge, 1=Lasso
}

# Hyperparameter grid for LightGBM
lgb_param_grid = {
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__n_estimators': [100, 500, 1000],
    'regressor__num_leaves': [31, 50],
    'regressor__min_child_weight': [1, 5]
}

# GLM hyperparameter tuning
glm_search = GridSearchCV(
    estimator=GLM_pipeline,
    param_grid=glm_limited_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',  # MSE
    n_jobs=-1
)

# GLM hyperparameter tuning for the limited feature set
glm_limited_search = GridSearchCV(
    estimator=GLM_pipeline,
    param_grid=glm_limited_param_grid,
    cv=5,
    scoring='neg_mean_squared_error',  # MSE
    n_jobs=-1
)

# LightGBM
LGBM_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(objective='regression', n_estimators=5000))
])

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
glm_search.fit(X_train, y_train)
print("Best GLM params:", glm_search.best_params_)

glm_limited_search.fit(X_train, y_train)
print("Best GLM params:", glm_search.best_params_)

# Fit LGBM with GridSearchCV
lgb_search.fit(X_train, y_train)
print("Best LGBM params:", lgb_search.best_params_)

print("Training complete.")

# ----------------------------
# Predictions
# ----------------------------
y_pred_GLM = glm_search.best_estimator_.predict(X_test)
y_pred_GLM_limited = glm_limited_search.best_estimator_.predict(X_test_limited)
y_pred_LGBM = lgb_search.best_estimator_.predict(X_test)

# ----------------------------
# Evaluate MSE and R^2
# ----------------------------
mse_glm = mean_squared_error(y_test, y_pred_GLM)
r2_glm = r2_score(y_test, y_pred_GLM)

mse_glml = mean_squared_error(y_test, y_pred_GLM_limited)
r2_glml = r2_score(y_test, y_pred_GLM_limited)

mse_lgbm = mean_squared_error(y_test, y_pred_LGBM)
r2_lgbm = r2_score(y_test, y_pred_LGBM)

print(f"GLM - MSE: {mse_glm:.2f}, R^2: {r2_glm:.3f}")
print(f"GLMlimited - MSE: {mse_glml:.2f}, R^2: {r2_glml:.3f}")
print(f"LGBM - MSE: {mse_lgbm:.2f}, R^2: {r2_lgbm:.3f}")