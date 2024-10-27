import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("ML/alldata_filtered.csv")


hit_data = df[df['hit'] == True]
non_hit_data = df[df['hit'] == False]

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

features = [
    'x', 'y', 'z', 'gx', 'gy', 'gz',
    'accel_magnitude', 'gyro_magnitude',
    'x_roll_avg', 'y_roll_avg', 'z_roll_avg',
    'accel_magnitude_roll_avg',
    'gx_roll_avg', 'gy_roll_avg', 'gz_roll_avg',
    'gyro_magnitude_roll_avg'
]


accel_count = 0
gyro_count = 0


for index, row in df.iterrows():
    if row['accel_magnitude'] > 5:
        accel_count += 1
    if row['gyro_magnitude'] > 5:
        gyro_count += 1


print(f"Accelerometer magnitude exceeded 5 a total of {accel_count} times.")
print(f"Gyroscope magnitude exceeded 5 a total of {gyro_count} times.")


plt.figure(figsize=(12, 6))


entry_order = np.arange(len(df))


plt.plot(entry_order, df['accel_magnitude'], label='Accelerometer Magnitude')
plt.plot(entry_order, df['gyro_magnitude'], label='Gyroscope Magnitude')


plt.fill_between(entry_order, df['accel_magnitude'].min(), df['accel_magnitude'].max(),
                 where=(df['hit'] == True), color='green', alpha=0.3, label='Hit = True')


plt.fill_between(entry_order, df['accel_magnitude'].min(), df['accel_magnitude'].max(),
                 where=(df['hit'] == False), color='red', alpha=0.3, label='Hit = False')


plt.legend()
plt.xlabel('Entry Order')
plt.ylabel('Magnitude')
plt.title('Accelerometer and Gyroscope Magnitude with Hit Status')
plt.show()
