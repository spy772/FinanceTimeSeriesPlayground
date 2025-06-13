import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def generate_variable_trend(trend_type, length=12, scale_range=(50, 10000), noise_level=0.1):
    # Base time array, not always uniformly spaced
    t = np.linspace(0, 1, length)
    
    # Start with a base trend function
    if trend_type == 'increasing':
        trend = np.power(t, random.uniform(0.8, 2.5))  # power curve
    elif trend_type == 'increasing_slow':
        trend = np.log1p(t * random.uniform(2, 10))  # slow log growth
    elif trend_type == 'neutral':
        trend = np.full(length, 0.5) + np.random.normal(0, 0.05, size=length)
    elif trend_type == 'decreasing_slow':
        trend = np.log1p((1 - t) * random.uniform(2, 10))
    elif trend_type == 'decreasing':
        trend = np.power((1 - t), random.uniform(0.8, 2.5))
    else:
        raise ValueError("Invalid trend type")

    # Normalize to 0-1
    trend = (trend - np.min(trend)) / (np.max(trend) - np.min(trend) + 1e-6)
    
    # Add a delay or acceleration in the trend's onset
    if trend_type != 'neutral':
        shift = random.randint(-2, 2)
        trend = np.roll(trend, shift)
        if shift > 0:
            trend[:shift] = trend[shift]
        elif shift < 0:
            trend[shift:] = trend[shift - 1]

    # Add jaggedness: localized jumps/dips
    jagged = np.copy(trend)
    for i in range(length):
        if random.random() < 0.3:  # 30% chance of local distortion
            jagged[i] += np.random.normal(0, 0.1)
    
    # Add sharp jumps or drops occasionally
    for _ in range(random.randint(0, 2)):
        jump_idx = random.randint(1, length - 2)
        jump_magnitude = np.random.normal(0, 0.2)
        jagged[jump_idx] += jump_magnitude

    # Rescale to target value range
    min_val = random.uniform(scale_range[0], scale_range[1] * 0.2)
    max_val = random.uniform(scale_range[1] * 0.5, scale_range[1])
    scaled = jagged * (max_val - min_val) + min_val

    # Final noise
    noise = np.random.normal(0, noise_level * (max_val - min_val), size=length)
    final_series = scaled + noise

    return final_series.tolist()

# Generate data for each category
categories = {
    'increasing': [],
    'increasing_slow': [],
    'neutral': [],
    'decreasing_slow': [],
    'decreasing': []
}

for _ in range(15):
    for cat in categories:
        categories[cat].append(generate_variable_trend(cat, noise_level=random.random()*0.15))

# Save data to CSV

# Flatten data into rows with category labels
all_series = []
for cat, series_list in categories.items():
    for series in series_list:
        row = {'category': cat}
        for i, value in enumerate(series):
            row[f't{i+1}'] = value
        all_series.append(row)

# Create DataFrame and save to CSV
df = pd.DataFrame(all_series)
df.to_csv("synthetic_timeseries_data.csv", index=False)

print("Saved to synthetic_timeseries_data.csv")

# Optional plot
for cat, series_list in categories.items():
    plt.figure(figsize=(10, 4))
    for series in series_list[:15]:
        plt.plot(series)
    plt.title(f"{cat} (sample of 5)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()
