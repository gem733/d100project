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

preprocessorlimited = ColumnTransformer(
    transformers=[
        ('log_numeric', LogTransformer(), ['budget']),
        ('num', StandardScaler(), ['runtime','year']),
        ('month', MonthToSeasonTransformer(month_column=month_feature), [month_feature])
    ]
)

# Create the model pipelines

# GLM
GLMmodel = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

GLMlimitedmodel = Pipeline(steps=[
    ('preprocessor', preprocessorlimited),
    ('regressor', LinearRegression())
])

# LightGBM
LGBMmodel = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', lgb.LGBMRegressor(objective='regression', metric='mse'))
])

# Train the models
GLMmodel.fit(X_train, y_train)
GLMlimitedmodel.fit(X_train_limited, y_train)
LGBMmodel.fit(X_train, y_train)

print("Training complete.")

# ----------------------------
# Predictions
# ----------------------------
y_pred_GLM = GLMmodel.predict(X_test)
y_pred_GLM_limited = GLMlimitedmodel.predict(X_test_limited)
y_pred_LGBM = LGBMmodel.predict(X_test)

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