import pandas as pd 
import numpy as np 

def add_store_info(train,stores):
    train = train.copy()
    return train.merge(stores, on="store_nbr", how="left") 

def add_oil_info(train, oil):
    train = train.copy()
    oil = oil.copy()

    train["date"] = pd.to_datetime(train["date"])
    oil["date"] = pd.to_datetime(oil["date"])

    train = train.merge(oil[["date", "dcoilwtico"]], on="date", how = "left")
    train["dcoilwtico"] = train["dcoilwtico"].ffill().bfill()

    return train

def add_holiday_binary(train,holidays_events):
    train = train.copy()
    holidays = holidays_events.copy()

    holidays["date"] = pd.to_datetime(holidays["date"])
    holidays = holidays[holidays["transferred"] == False]

    real_holidays= holidays[holidays["type"]=="Holiday"].copy()
    holiday_national = real_holidays[real_holidays["locale"]=="National"].copy()
    holiday_national["holiday_national_binary"] = 1

    holiday_national = (holiday_national.groupby("date", as_index=False)["holiday_national_binary"].max())

    train["date"] = pd.to_datetime(train["date"])
    train = train.merge(holiday_national, on="date", how="left")
    train["holiday_national_binary"] = train["holiday_national_binary"].fillna(0).astype(int)

    return train

def add_event_flags(train, holidays_events):
    train = train.copy()
    events = holidays_events.copy()

    events= events[events["type"]=="Event"].copy()
    events["date"] = pd.to_datetime(events["date"])

    events["description"] = (
        events["description"]
        .astype(str)
        .str.lower()
        .str.strip()
    )
    events["is_black_friday"] = events["description"].str.contains("black friday").astype(int)
    events["is_cyber_monday"] = events["description"].str.contains("cyber monday").astype(int)
    events["is_mothers_day"] = events["description"].str.contains("dia de la madre").astype(int)
    events["is_earthquake"] = events["description"].str.contains("terremoto manabi").astype(int) 
    events["is_world_cup"] = events["description"].str.contains("mundial").astype(int)

    event_cols = [
        "is_black_friday",
        "is_cyber_monday",
        "is_mothers_day",
        "is_earthquake",
        "is_world_cup"
    ]

    event_daily = events.groupby("date", as_index = "False")[event_cols].max()
    train["date"] = pd.to_datetime(train["date"])
    train = train.merge(event_daily, on="date",how="left")
    train[event_cols] = train[event_cols].fillna(0).astype(int)

    return train

def add_calender_features(train):
    train = train.copy()
    train["date"] = pd.to_datetime(train["date"])

    train["year"] = train["date"].dt.year 
    train["month"] = train["date"].dt.month
    train["day"] = train["date"].dt.day
    train["dayofweek"] = train["date"].dt.dayofweek 
    train["is_weekend"] = train["dayofweek"].isin([5,6]).astype(int)
    train["is_month_start"] = train["date"].dt.is_month_start.astype(int)
    train["is_month_end"] = train["date"].dt.is_month_end.astype(int)

    return train

def add_lag_rolling_features(train):
    train = train.copy()
    train= train.sort_values(["store_nbr", "family", "date"])
    grouped = train.groupby(["store_nbr", "family"])["sales"]

    train["lag_1"] = grouped.shift(1) 
    train["lag_7"] = grouped.shift(7)
    train["lag_14"] = grouped.shift(14)

    train["rolling_mean_7"] = grouped.transform(lambda x: x.shift(1).rolling(7).mean())
    train["rolling_mean_14"] = grouped.transform(lambda x: x.shift(1).rolling(14).mean())
    train["rolling_mean_30"] = grouped.transform(lambda x: x.shift(1).rolling(30).mean())
    train["rolling_std_7"] = grouped.transform(lambda x: x.shift(1).rolling(7).std())
    train["promo_last_7"] = grouped.transform(lambda x: x.shift(1).rolling(7).sum())
    
    if "dcoilwtico" in train.columns:
        train["oil_change"] = train["dcoilwtico"].diff() 
    else:
        train["oil_change"] = 0

    feature_cols = [
        "lag_1", "lag_7", "lag_14",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_7", "promo_last_7", "oil_change"
    ]

    train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], 0)
    train[feature_cols] = train[feature_cols].fillna(0)
    
    return train

def build_feature_dataframe(train, stores, oil, holidays_events):
    df = add_store_info(train, stores)
    df = add_oil_info(df, oil)
    df = add_holiday_binary(df, holidays_events)
    df = add_event_flags(df, holidays_events)
    df = add_calender_features(df)
    df = add_lag_rolling_features(df)
    return df









