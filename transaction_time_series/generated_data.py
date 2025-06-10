# Need to create at least one example of the 5 following time series types:
# General increase
# Increase that slows down (could mean just flattening out, or going down near the end)
# General decrease
# Decrease that slows down (same as increase, but going up)
# Neutral
#
# About 12 timepoints per observation
import numpy as np
import pandas as pd


up_1 = [16.8, 21.3, 25.1, 29.2, 33.3, 35.8, 44.8, 50.7, 51.4, 55.7, 60.2, 65.3]
up_2 = [26.1, 27.2, 33.5, 34.6, 38.7, 39.3, 44.2, 47.3, 48.2, 51.7, 54.3, 57.1]
up_3 = [13.0, 17.5, 24.1, 28.3, 32.7, 36.1, 40.9, 44.1, 46.2, 51.5, 54.9, 59.7]
up_4 = [11.2, 13.3, 21.3, 22.7, 25.0, 29.6, 31.6, 39.0, 44.0, 46.2, 50.5, 58.8]
up_5 = [22.3, 25.6, 29.5, 32.2, 35.6, 36.8, 39.4, 42.6, 46.2, 48.2, 51.3, 55.8]
up_6 = [20.5, 20.9, 22.0, 25.3, 26.8, 29.5, 36.8, 39.7, 46.8, 50.3, 52.6, 58.4]

up_slowed_1 = [19.2, 20.9, 22.0, 24.0, 28.3, 30.0, 35.5, 37.3, 40.2, 44.3, 50.3, 52.6]
up_slowed_2 = [22.6, 20.7, 21.4, 22.4, 30.9, 32.3, 36.3, 37.2, 41.7, 42.4, 43.9, 44.1]
up_slowed_3 = [20.5, 21.7, 24.1, 26.0, 28.3, 33.2, 35.5, 37.9, 40.2, 41.0, 39.3, 36.2]
up_slowed_4 = [20.5, 21.2, 23.0, 25.4, 28.1, 30.3, 32.1, 30.4, 28.6, 23.8, 21.6, 20.7]
up_slowed_5 = [21.0, 22.4, 24.8, 29.1, 31.5, 34.0, 38.7, 40.2, 41.6, 39.5, 38.7, 37.7]
up_slowed_6 = [20.5, 21.7, 22.0, 25.3, 29.5, 32.3, 36.8, 39.7, 46.8, 48.6, 49.1, 49.3]

down_1 = [60.0, 58.5, 56.9, 55.2, 53.1, 48.9, 43.6, 41.0, 38.2, 35.5, 27.1, 24.6]
down_2 = [59.0, 57.3, 54.5, 53.1, 49.5, 44.5, 39.2, 37.8, 34.4, 30.2, 22.9, 21.1]
down_3 = [59.5, 57.9, 56.2, 54.4, 51.9, 47.3, 45.0, 42.5, 36.6, 34.0, 25.8, 22.9]
down_4 = [62.0, 60.7, 58.4, 56.1, 53.8, 48.9, 44.3, 41.8, 38.7, 35.6, 27.6, 24.1]
down_5 = [65.3, 57.4, 54.8, 52.3, 49.7, 43.0, 39.2, 39.1, 36.7, 31.5, 23.7, 20.6]
down_6 = [60.0, 57.4, 54.0, 51.4, 47.8, 40.7, 35.6, 33.6, 30.2, 26.6, 22.6, 20.9]

down_slowed_1 = [51.9, 47.3, 47.9, 43.2, 43.4, 41.1, 35.9, 33.7, 33.9, 28.6, 13.7, 10.0]
down_slowed_2 = [54.4, 47.0, 42.4, 41.4, 36.6, 35.6, 36.2, 33.4, 30.8, 16.8, 10.0, 10.8]
down_slowed_3 = [52.5, 48.0, 43.3, 40.1, 38.3, 38.0, 35.7, 33.7, 31.6, 34.1, 12.1, 11.8]
down_slowed_4 = [49.5, 47.1, 44.2, 41.1, 44.5, 39.3, 38.0, 35.3, 35.8, 37.3, 12.2, 10.0]
down_slowed_5 = [50.2, 48.5, 42.8, 41.8, 40.0, 35.8, 38.2, 35.1, 31.5, 30.5, 15.4, 10.0]
down_slowed_6 = [51.9, 47.3, 47.9, 43.2, 43.4, 41.1, 35.9, 33.7, 33.9, 28.6, 13.7, 10.0]

neutral_1 = [60.0, 64.3, 55.9, 62.0, 66.7, 59.2, 61.0, 57.9, 60.1, 62.3, 66.9, 67.4]
neutral_2 = [20.0, 24.6, 21.1, 22.5, 18.7, 29.4, 24.1, 19.8, 25.2, 28.0, 20.2, 22.8]
neutral_3 = [35.0, 33.4, 38.1, 34.0, 29.1, 36.7, 30.4, 32.9, 35.6, 31.5, 36.1, 32.0]
neutral_4 = [12.0, 18.2, 15.7, 13.8, 12.5, 17.4, 14.0, 17.7, 14.4, 16.2, 17.1, 13.5]
neutral_5 = [68.0, 63.5, 70.0, 66.7, 61.2, 64.0, 70.0, 67.6, 68.9, 66.1, 65.5, 70.0]
neutral_6 = [44.3, 54.3, 50.3, 48.5, 33.8, 38.6, 45.1, 66.0, 47.6, 54.0, 49.3, 52.5]

# Aggregate the datasets
up_arr = [up_1, up_2, up_3, up_4, up_5, up_6]
up_slow_arr = [up_slowed_1, up_slowed_2, up_slowed_3, up_slowed_4, up_slowed_5, up_slowed_6]
down_arr = [down_1, down_2, down_3, down_4, down_5, down_6]
down_slow_arr = [down_slowed_1, down_slowed_2, down_slowed_3, down_slowed_4, down_slowed_5, down_slowed_6]
neutral_arr = [neutral_1, neutral_2, neutral_3, neutral_4, neutral_5, neutral_6]

X = [up_1, up_2, up_3, up_4, up_5, up_6, 
     up_slowed_1, up_slowed_2, up_slowed_3, up_slowed_4, up_slowed_5, up_slowed_6, 
     neutral_1, neutral_2, neutral_3, neutral_4, neutral_5, neutral_6, 
     down_slowed_1, down_slowed_2, down_slowed_3, down_slowed_4, down_slowed_5, down_slowed_6, 
     down_1, down_2, down_3, down_4, down_5, down_6] # Note the ordering (most extreme on edges)
# Classifiers should be 0, 1, 2, 3, 4 in for each arr-type above
# Create classifiers for each one
y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]

def load_timeseries_data(as_numpy: bool) -> list[list[float]] | np.ndarray:
    """
    Returns X, y either as Python array or numpy array
    """

    # Validate equal length first
    if len(X) != len(y):
        print(len(X))
        print(len(y))
        raise RuntimeError("X and y should have same length")
    
    if as_numpy:
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int32)

        return X_np, y_np

    return X, y
