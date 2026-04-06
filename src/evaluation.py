import  numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def safe_mape(y_true, y_pred,eps=1e-8): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > eps 
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))*100


def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae= mean_absolute_error(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)

    return{
        "rmse" : rmse,
        "mae" : mae,
        "mape" :mape
    }

def build_results_df(meta_df, y_true, y_pred):
    results_df = meta_df.copy()
    results_df["y_true"] = np.array(y_true)
    results_df["y_preds"] = np.array(y_pred)
    results_df["residual"] = results_df["y_true"] - results_df["y_preds"]
    results_df["abs_error"] = np.abs(results_df["residual"])

    mask = np.abs(results_df["y_true"]) > 1e-8
    results_df["ape"] = np.nan
    results_df.loc[mask, "ape"] = (
        np.abs(results_df.loc[mask, "y_true"]- results_df.loc[mask, "y_preds"]) /
        np.abs(results_df.loc[mask, "y_true"]) * 100
    )

    return results_df

def group_metrics(results_df, group_col):
    grouped = (
    results_df.groupby(group_col).apply(lambda g: pd.Series({
        "MAE": mean_absolute_error(g["y_true"], g["y_preds"]),
        "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_preds"])),
        "MAPE": np.nanmean(g["ape"])
        }), include_groups= False).sort_values("MAE", ascending = False)
    )
    return grouped

def evaluate_predictions(y_true, y_pred):
    metrics = calculate_metrics(y_true, y_pred)
    return metrics

