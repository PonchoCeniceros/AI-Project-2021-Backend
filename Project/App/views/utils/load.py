import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = os.path.join(os.path.join(BASE_DIR, "db"), "base.pkl")

import pickle
import pandas as pd

def load():
    """ returns base dataset for training

    Returns:
        pandas.DataFrame: base dataset
    """
    return pd.read_pickle(DATA_DIR)    