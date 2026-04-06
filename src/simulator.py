import numpy as np
import pandas as pd


def get_sequence_features():
    return [
        "lag_sales",
        "onpromotion",
        "dcoilwtico",
        "is_weekend",
        "holiday_national_binary",
        "is_black_friday",
        "is_cyber_monday",
        "is_mothers_day",
        "is_earthquake",
        "is_world_cup"
    ]


def build_seed_dataframe(
    train_merged,
    target_store,
    target_family,
    window=14,
    sequence_features=None
):
    if sequence_features is None:
        sequence_features = get_sequence_features()

    seed_df = (
        train_merged[
            (train_merged["store_nbr"] == target_store) &
            (train_merged["family"] == target_family)
        ]
        .sort_values("date")
        .tail(window)
        .copy()
    )

    for col in sequence_features:
        if col not in seed_df.columns:
            seed_df[col] = 0

    seed_df[sequence_features] = (
        seed_df[sequence_features]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    return seed_df


def build_seed_values(seed_df, sequence_features=None):
    if sequence_features is None:
        sequence_features = get_sequence_features()

    return seed_df[sequence_features].values.astype(np.float32)


def simulate_scenario(seed_values, promo_val, future_days, model, scaler, sequence_features):
    window_buf = seed_values.copy()
    preds = []

    lag_idx = sequence_features.index("lag_sales")
    promo_idx = sequence_features.index("onpromotion")

    for _ in range(future_days):
        window_scaled = scaler.transform(window_buf)
        model_input = window_scaled[np.newaxis, :, :]

        pred = model.predict(model_input, verbose=0)[0][0]
        preds.append(float(pred))

        new_row = window_buf[-1].copy()
        new_row[lag_idx] = pred
        new_row[promo_idx] = promo_val

        window_buf = np.vstack([window_buf[1:], new_row])

    return np.array(preds)


def build_whatif_results(seed_df, preds_no_promo, preds_with_promo, future_days):
    preds_no_promo = np.array(preds_no_promo)
    preds_with_promo = np.array(preds_with_promo)

    diff = preds_with_promo - preds_no_promo
    uplift = diff / (preds_no_promo + 1e-8) * 100

    future_dates = pd.date_range(
        start=seed_df["date"].max() + pd.Timedelta(days=1),
        periods=future_days
    )

    results_whatif = pd.DataFrame({
        "date": future_dates,
        "no_promo": preds_no_promo.round(2),
        "with_promo": preds_with_promo.round(2),
        "diff": diff.round(2),
        "uplift_pct": uplift.round(2)
    })

    return results_whatif


def run_whatif_scenario(
    train_merged,
    target_store,
    target_family,
    model,
    scaler,
    future_days=15,
    window=14,
    promo_no=0,
    promo_yes=1,
    sequence_features=None
):
    if sequence_features is None:
        sequence_features = get_sequence_features()

    seed_df = build_seed_dataframe(
        train_merged=train_merged,
        target_store=target_store,
        target_family=target_family,
        window=window,
        sequence_features=sequence_features
    )

    seed_values = build_seed_values(seed_df, sequence_features)

    preds_no_promo = simulate_scenario(
        seed_values=seed_values,
        promo_val=promo_no,
        future_days=future_days,
        model=model,
        scaler=scaler,
        sequence_features=sequence_features
    )

    preds_with_promo = simulate_scenario(
        seed_values=seed_values,
        promo_val=promo_yes,
        future_days=future_days,
        model=model,
        scaler=scaler,
        sequence_features=sequence_features
    )

    results_whatif = build_whatif_results(
        seed_df=seed_df,
        preds_no_promo=preds_no_promo,
        preds_with_promo=preds_with_promo,
        future_days=future_days
    )

    return seed_df, results_whatif