import numpy as np
import pandas as pd

np.random.seed(42)
length = 100
x = np.arange(length)

# Helper to scale any series to roughly [10, 3000]
def scale_to_range(data, min_val=10, max_val=3000):
    data_min = np.min(data)
    data_max = np.max(data)
    scaled = (data - data_min) / (data_max - data_min)  # normalize to [0, 1]
    return scaled * (max_val - min_val) + min_val

def generate_linear_trend(slope=0.1, intercept=0, noise_level=1.0):
    noise = np.random.normal(0, noise_level, length)
    return slope * x + intercept + noise

def generate_seasonal_trend(slope=0.1, intercept=0, season_amplitude=10, season_period=20, noise_level=1.0):
    seasonal = season_amplitude * np.sin(2 * np.pi * x / season_period)
    noise = np.random.normal(0, noise_level, length)
    return slope * x + intercept + seasonal + noise

# Generate and scale each series
upward_linear = scale_to_range(generate_linear_trend(slope=0.3, intercept=5, noise_level=2))
upward_seasonal = scale_to_range(generate_seasonal_trend(slope=0.3, intercept=5, season_amplitude=5, season_period=20, noise_level=2))

neutral_linear = scale_to_range(generate_linear_trend(slope=0.0, intercept=50, noise_level=2))
neutral_seasonal = scale_to_range(generate_seasonal_trend(slope=0.0, intercept=50, season_amplitude=5, season_period=20, noise_level=2))

downward_linear = scale_to_range(generate_linear_trend(slope=-0.3, intercept=80, noise_level=2))
downward_seasonal = scale_to_range(generate_seasonal_trend(slope=-0.3, intercept=80, season_amplitude=5, season_period=20, noise_level=2))

# Bundle all into dictionary
time_series_data = {
    "upward_linear": upward_linear,
    "upward_seasonal": upward_seasonal,
    "neutral_linear": neutral_linear,
    "neutral_seasonal": neutral_seasonal,
    "downward_linear": downward_linear,
    "downward_seasonal": downward_seasonal
}


# Convert the dictionary to a DataFrame
df = pd.DataFrame(time_series_data)

# Transpose so each row is a series, and each column a timestep
df = df.T

# Optional: name columns as t0, t1, ..., t99
df.columns = [f"t{i}" for i in range(length)]

# Save to CSV
df.to_csv("data_generation/time_series_data.csv", index_label="series_id")

print("Saved to time_series_data.csv")