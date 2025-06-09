import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)

def generate_series(trend_type, range_type, length=12):
    """Generate a time series given the trend type and range scale."""
    if range_type == "large":
        base = np.random.uniform(1000, 10000)
        range_span = np.random.uniform(5000, 9000)
    elif range_type == "small":
        base = np.random.uniform(100, 500)
        range_span = np.random.uniform(100, 200)
    else:  # tight range
        base = np.random.uniform(2000, 4000)
        range_span = np.random.uniform(50, 100)

    trend = np.linspace(0, range_span, length)
    noise = np.random.normal(0, range_span * 0.05, length)

    if trend_type == "increasing":
        series = base + trend + noise
    elif trend_type == "up_slowed":
        slow_curve = np.power(np.linspace(0, 1, length), 0.7) * range_span
        series = base + slow_curve + noise
    elif trend_type == "neutral":
        neutral_fluctuation = np.random.uniform(-1, 1, length).cumsum()
        neutral_fluctuation = neutral_fluctuation / max(abs(neutral_fluctuation)) * (range_span / 2)
        series = base + neutral_fluctuation + noise
    elif trend_type == "down_slowed":
        slow_down = np.power(np.linspace(1, 0, length), 0.7) * range_span
        series = base + slow_down + noise
    elif trend_type == "down":
        series = base + (-trend) + noise
    else:
        raise ValueError("Unknown trend type")

    return np.round(series, 2).tolist()

# Define trend types and range types
trend_types = ["increasing", "up_slowed", "neutral", "down_slowed", "down"]
range_types = ["large", "small", "tight"]

categories = {trend: [] for trend in trend_types}

# Generate 15 per trend: 5 for each range category
for trend in trend_types:
    for range_type in range_types:
        for _ in range(5):
            series = generate_series(trend, range_type)
            categories[trend].append(series)

# Prepare CSV output
all_rows = []
for trend, series_list in categories.items():
    for series in series_list:
        row = {"category": trend}
        for i, val in enumerate(series):
            row[f"t{i+1}"] = val
        all_rows.append(row)

df = pd.DataFrame(all_rows)
df.to_csv("synthetic_timeseries_balanced_ranges.csv", index=False)
print("Saved to synthetic_timeseries_balanced_ranges.csv")

# Optional plot
for cat, series_list in categories.items():
    plt.figure(figsize=(10, 4))
    for series in series_list[:5]:
        plt.plot(series)
    plt.title(f"{cat} (sample of 5)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
