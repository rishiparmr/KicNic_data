import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('testdata.csv')


initial_time = df['time'].iloc[0]  
df['time'] = df['time'] - initial_time  






z_scores = np.abs(stats.zscore(df[['x', 'y', 'z', 'gx', 'gy', 'gz']]))
df = df[(z_scores < 3).all(axis=1)]  


scaler = StandardScaler()
df[['x', 'y', 'z', 'gx', 'gy', 'gz']] = scaler.fit_transform(df[['x', 'y', 'z', 'gx', 'gy', 'gz']])




df['accel_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
df['gyro_magnitude'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)


window_size = 5


df['x_roll_avg'] = df['x'].rolling(window=window_size).mean()
df['y_roll_avg'] = df['y'].rolling(window=window_size).mean()
df['z_roll_avg'] = df['z'].rolling(window=window_size).mean()
df['accel_magnitude_roll_avg'] = df['accel_magnitude'].rolling(window=window_size).mean()


df['gx_roll_avg'] = df['gx'].rolling(window=window_size).mean()
df['gy_roll_avg'] = df['gy'].rolling(window=window_size).mean()
df['gz_roll_avg'] = df['gz'].rolling(window=window_size).mean()
df['gyro_magnitude_roll_avg'] = df['gyro_magnitude'].rolling(window=window_size).mean()


df = df.dropna()




print(df['hit'].value_counts())

hit_count = 0
not_hit_count = 0
in_hit_block = False

for is_hit in df['hit']:
    if is_hit:
        
        if not in_hit_block:
            hit_count += 1
            in_hit_block = True
    else:
        
        if in_hit_block:
            not_hit_count += 1
            in_hit_block = False

if not in_hit_block:
    not_hit_count += 1

print(f"Number of 'hit' events (blocks of True): {hit_count}")
print(f"Number of 'not hit' events (blocks of False): {not_hit_count}")


df.to_csv('test_data_clean.csv', index=False)

print("Data cleaned and saved to 'cleaned_data.csv'")
