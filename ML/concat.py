import pandas as pd

# Load the CSV file
df = pd.read_csv("/Users/rishi/VapeDetectorModel/ML/alldata.csv")

# Define the ranges to remove
start_remove_1 = 41390
end_remove_1 = 64000
start_remove_2 = 74990

# Create a mask to keep only rows outside the specified ranges
mask = pd.Series([True] * len(df))
mask[start_remove_1:end_remove_1 + 1] = False  # Remove entries from 41390 to 64000
mask[start_remove_2 + 1:] = False  # Remove all entries after 74990

# Filter the DataFrame
df_filtered = df[mask]

# Save the modified DataFrame to a new CSV file
output_file = "/Users/rishi/VapeDetectorModel/ML/alldata_filtered.csv"
df_filtered.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
