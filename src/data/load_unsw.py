import pandas as pd


def load_unsw_csv(path: str) -> pd.DataFrame:
    """
    Load UNSW-NB15 CSV and normalize column names.
    """
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df
