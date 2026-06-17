# src/evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_forecast(y_true, y_pred, model_name="Model"):
    """
    Calculates and prints classic predictive metrics comparing actuals vs forecasts.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional accuracy (did we get the sign of the next move right?)
    actual_dir = np.diff(y_true) > 0
    pred_dir   = np.diff(y_pred) > 0
    da         = np.mean(actual_dir == pred_dir) * 100

    print(f"\n=== {model_name} ===")
    print(f"  RMSE              : {rmse:.2f} EUR/tCO2")
    print(f"  MAE               : {mae:.2f} EUR/tCO2")
    print(f"  MAPE              : {mape:.2f}%")
    print(f"  Directional Acc.  : {da:.1f}%")
    
    return {"model": model_name, "RMSE": rmse, "MAE": mae,
            "MAPE": mape, "Dir_Acc": da}


def run_full_benchmark(y_test, oos_preds):
    """
    Wraps the baseline benchmarks so they don't break notebook imports.
    Call this inside your notebook to get the final comparative table!
    """
    # Naive benchmark: predict tomorrow = today (random walk)
    naive_preds = y_test.values[:-1]
    naive_true  = y_test.values[1:]

    results = []
    results.append(evaluate_forecast(naive_true, naive_preds, "Naive Random Walk"))
    results.append(evaluate_forecast(y_test, oos_preds, "XGBoost"))
    # Add VAR and ARIMAX results similarly as you expand your pipeline

    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))
    return results_df