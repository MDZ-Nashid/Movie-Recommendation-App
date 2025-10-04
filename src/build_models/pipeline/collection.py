import pandas as pd

from config.model_config import model_settings


def load_data(path=model_settings.data_file):
    return pd.read_csv(path)