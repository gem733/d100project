import sys
from pathlib import Path

# Set up project root to allow me to import the model for evaluation
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from d100project.data._create_sample_split import create_sample_split
from d100project.data._load_cleaned_parquet import load_cleaned_parquet
from d100project.preprocessing._log_transformer import LogTransformer
from d100project.preprocessing._month_to_season import MonthToSeasonTransformer
from d100project.preprocessing._one_hot_encode import ListOneHotEncoder

# Load cleaned data
df = load_cleaned_parquet()

# Create train/test split column
df = create_sample_split(df, id_column="id", training_frac=0.8)

# create a column for ln(revenue) to allow for evaluation later
df['ln_revenue'] = np.log1p(df['revenue'])

# training and testing dataframes
df_train = df[df["sample"] == "train"].copy()
df_test = df[df["sample"] == "test"].copy()


# Define features and target

target = "revenue"

numeric_features = ["budget", "runtime", "year"]
list_features = ["genres_list"]  # OHE only for genres
high_cardinality = ["original_language"]  # target encoding
count_features = ["n_production_companies", "n_production_countries", "n_spoken_languages"]
month_feature = "month"
indicators = ['runtime_was_missing', 'budget_was_missing']

all_features = numeric_features + list_features + [month_feature] + high_cardinality + count_features + indicators


# split into X and y, and train/test

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
te_transformer = Pipeline([
    ("te", TargetEncoder(smoothing=10, min_samples_leaf=20))
])

# Preprocessors

preprocessor = ColumnTransformer(
    transformers=[
        ('log_numeric', LogTransformer(), ['budget']),
        ('num', StandardScaler(), ['runtime','year']),
        ('list', ListOneHotEncoder(columns=list_features), list_features),
        ('month', MonthToSeasonTransformer(month_column=month_feature), [month_feature]),
        ("indicators", "passthrough", indicators)
    ]
)

preprocessor_lgbm = ColumnTransformer(
    transformers=[
        ("log_budget", LogTransformer(), ["budget"]),
        ("num", "passthrough", ["runtime", "year"] + count_features),
        ("te", TargetEncoder(smoothing=10, min_samples_leaf=20), ["original_language"]),
        ("genres", ListOneHotEncoder(columns=list_features), ["genres_list"]),
        ("month", MonthToSeasonTransformer(month_column=month_feature), [month_feature]),
        ("indicators", "passthrough", indicators)
    ]
)


# fit preprocessors and transform data

preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

X_train_lgbm = preprocessor_lgbm.fit_transform(X_train, y_train)
X_test_lgbm = preprocessor_lgbm.transform(X_test)

# Create the model pipelines

# GLM
GLM_pipeline = ElasticNet(max_iter=10000)

# Hyperparameter grid for GLM
glm_param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0] 
}


# Hyperparameter grid for LightGBM
lgb_param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [5000, 10000],
    'num_leaves': [15, 31],
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
LGBM_pipeline = lgb.LGBMRegressor(objective='regression', n_estimators=5000, random_state=42)

# LGBM hyperparameter tuning

lgb_search = GridSearchCV(
    estimator=LGBM_pipeline,
    param_grid=lgb_param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
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
        early_stopping(stopping_rounds=20),
        log_evaluation(100)
    ]
)
print("Best LGBM params:", lgb_search.best_params_)


# Predictions

y_pred_GLM = glm_search.best_estimator_.predict(X_test_transformed)
y_pred_LGBM = lgb_search.best_estimator_.predict(X_test_lgbm)


# Evaluate MSE and R^2

mse_glm = mean_squared_error(y_test, y_pred_GLM)
r2_glm = r2_score(y_test, y_pred_GLM)


mse_lgbm = mean_squared_error(y_test, y_pred_LGBM)
r2_lgbm = r2_score(y_test, y_pred_LGBM)

print(f"GLM - MSE: {mse_glm:.2f}, R^2: {r2_glm:.3f}")
print(f"LGBM - MSE: {mse_lgbm:.2f}, R^2: {r2_lgbm:.3f}")

# hyperparameters

def report_search_results(name, search):
    print(f"\n{name} RESULTS")
    print("-" * 40)
    print("Best hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV MSE: {-search.best_score_:.4f}")

report_search_results("GLM", glm_search)
report_search_results("LGBM", lgb_search)


# Save or replace predictions in test DataFrame

if 'pred_glm' in df_test.columns:
    df_test['pred_glm'] = y_pred_GLM  # replace existing
else:
    df_test['pred_glm'] = y_pred_GLM  # create new column

if 'pred_lgbm' in df_test.columns:
    df_test['pred_lgbm'] = y_pred_LGBM
else:
    df_test['pred_lgbm'] = y_pred_LGBM


# Save the test DataFrame with predictions for evaluation

project_root = Path(__file__).resolve().parents[1]
output_path = project_root / "d100project" / "evaluation" / "df_test_with_predictions.parquet"
df_test.to_parquet(output_path, index=False)
print(f"Saved test set with predictions to {output_path}")


# Compute and save SHAP values

project_root = Path(__file__).resolve().parents[1]
shap_output_folder = project_root / "d100project" / "evaluation"
shap_output_folder.mkdir(exist_ok=True, parents=True)


# SHAP values computation for GLM

explainer_glm = shap.LinearExplainer(
    glm_search.best_estimator_,
    X_train_transformed,
    feature_perturbation="interventional"
)

# Compute SHAP values for the test set

shap_values_glm = explainer_glm.shap_values(X_test_transformed)


# Get feature names from the preprocessor so we can save them alongside SHAP values

feature_names_glm = preprocessor.get_feature_names_out()

expected_value_glm = explainer_glm.expected_value
# Save SHAP values along with feature names
np.save(
    shap_output_folder / "shap_glm_bundle.npy",
    {
        "shap_values": shap_values_glm,
        "expected_value": expected_value_glm,
        "feature_names": feature_names_glm,
        "X_test": X_test_transformed
    },
    allow_pickle=True
)

print("GLM SHAP values with feature names saved.")

explainer_lgbm = shap.TreeExplainer(lgb_search.best_estimator_)

shap_values_lgbm = explainer_lgbm.shap_values(X_test_lgbm)

feature_names_lgbm = preprocessor_lgbm.get_feature_names_out()

expected_value_lgbm = explainer_lgbm.expected_value

np.save(
    shap_output_folder / "shap_lgbm_bundle.npy",
    {
        "shap_values": shap_values_lgbm,
        "expected_value": expected_value_lgbm,
        "feature_names": feature_names_lgbm,
        "X_test": X_test_lgbm
    },
    allow_pickle=True
)
print("LGBM SHAP values with feature names saved.")


# Save the trained models and preprocessors for PDPs
project_root = Path(__file__).resolve().parents[1]
evaluation_dir = project_root / "d100project" / "evaluation"
evaluation_dir.mkdir(parents=True, exist_ok=True)

dump(lgb_search.best_estimator_, evaluation_dir / "lgbm_model.joblib")
dump(preprocessor_lgbm, evaluation_dir / "lgbm_preprocessor.joblib")

print("Saved LGBM model and preprocessor.")
