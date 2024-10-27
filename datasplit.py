import pandas as pd


df = pd.read_csv("cleaned_data6.csv")


consecutive_blocks = []


current_block = []
previous_value = None


for index, row in df.iterrows():
    if previous_value is None:
        previous_value = row['hit']

    
    if row['hit'] == previous_value:
        current_block.append(row)
    else:
        
        consecutive_blocks.append((previous_value, pd.DataFrame(current_block)))
        current_block = [row]
        previous_value = row['hit']


if current_block:
    consecutive_blocks.append((previous_value, pd.DataFrame(current_block)))


hit_count = 1
nothit_count = 1

for is_hit, block in consecutive_blocks:
    if is_hit:
        filename = f"hit{hit_count}.csv"
        hit_count += 1
    else:
        filename = f"nothit{nothit_count}.csv"
        nothit_count += 1
    
    
    block.to_csv(filename, index=False)
    print(f"Saved {filename}")
