import pandas as pd
from pathlib import Path

def get_data_dir():
    return Path(__file__).resolve().parent.parent/ "data"

def load_raw_data():
    data_dir = get_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    stores = pd.read_csv(data_dir / "stores.csv")
    oil = pd.read_csv(data_dir / "oil.csv")
    holidays_events = pd.read_csv(data_dir / "holidays_events.csv")
    transactions = pd.read_csv(data_dir / "transactions.csv")

    return {
        "train" :train,
        "test" : test,
        "stores" :stores,
        "oil" : oil,
        "holidays_events" :holidays_events,
        "transactions" : transactions,
    }

def parse_date_column(df, col="date"):
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_datetime(df[col])
    return df

def prepare_basic_dates(data_dict):
    prepared = {}
    for name, df in data_dict.items():
        if "date" in df.columns:
            prepared[name] = parse_date_column(df, "date")
        else:
            prepared[name] = df.copy()
    return prepared


