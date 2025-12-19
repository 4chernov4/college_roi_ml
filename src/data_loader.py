import pandas as pd
from config import (
    RAW_DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    META_COLUMNS
)

def load_data():
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN] + META_COLUMNS

    df = pd.read_csv(
        RAW_DATA_PATH,
        usecols=required_columns,
        low_memory=False
    )

    return df
