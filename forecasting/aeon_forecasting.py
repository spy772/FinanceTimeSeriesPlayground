from aeon.forecasting import RegressionForecaster
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from data_generation.generated_data import load_new_timeseries_data, load_original_timeseries_data 
import numpy as np
import pandas as pd
import tensorflow as tf

def main():
    df = pd.read_csv("data_generation/time_series_data.csv")
    df = df.drop("series_id", axis='columns')
    data = df.values.astype(np.float32)

    X_norm = normalize(data, norm="l2") # use when desired

    forecaster = RegressionForecaster(window=12, horizon=2)
    print(forecaster.forecast(X_norm[0])) # Choose first sample

    # Will not be accurate at all because the y values aren't forecast values...
    