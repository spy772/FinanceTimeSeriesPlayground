import pandas as pd

# Load your CSV file
df = pd.read_csv("synthetic_timeseries_data.csv")

# Initialize start index per category
start_index = 7
category_counters = {cat: start_index for cat in df['category'].unique()}

# Store all variable definitions
variable_definitions = []

# Iterate through the dataframe
for _, row in df.iterrows():
    category = row['category']
    index = category_counters[category]
    values = row[1:].tolist()  # All values except the category
    var_name = f"{category}_{index}"
    value_list = ", ".join(f"{v:.2f}" for v in values)
    variable_definitions.append(f"{var_name} = [{value_list}]")
    category_counters[category] += 1

# Print all variable definitions (or save to file)
for line in variable_definitions:
    print(line)

# Optional: Save to a .py file
with open("from_csv.py", "w") as f:
    f.write("\n".join(variable_definitions))
