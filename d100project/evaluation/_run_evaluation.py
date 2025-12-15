from _evaluate_predictions import evaluate_predictions
from _load_predictions import load_predict_parquet
import pandas as pd

# This script runs evaluation on prediction results from parquet file

df = load_predict_parquet()

# Evaluate GLM
eval_glm = evaluate_predictions(
    df,
    outcome_column='ln_revenue',
    preds_column='pred_glm'
)
print("GLM evaluation metrics:\n", eval_glm)

# Evaluate LGBM
eval_lgbm = evaluate_predictions(
    df,
    outcome_column='ln_revenue',
    preds_column='pred_lgbm'
)
print("LGBM evaluation metrics:\n", eval_lgbm)
