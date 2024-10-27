import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


data = pd.read_csv('cleaned_data5.csv')


accel_magnitude = data['accel_magitude'].values
gyro_magnitude = data['gyro_magnitude'].values
time = data.index  


accel_threshold = 5  
gyro_threshold = 5
pair_threshold = 20  
min_hit_duration = 0  
min_hit_separation = 5  


accel_peaks, _ = find_peaks(accel_magnitude, height=accel_threshold, distance=50)
gyro_peaks, _ = find_peaks(gyro_magnitude, height=gyro_threshold, distance=50)


hit_intervals = []
i, j = 0, 0


while i < len(accel_peaks) and j < len(gyro_peaks):
    if abs(accel_peaks[i] - gyro_peaks[j]) <= pair_threshold:
        start = min(accel_peaks[i], gyro_peaks[j])
        end = max(accel_peaks[i], gyro_peaks[j])
        
        
        if (end - start) >= min_hit_duration:
            
            if not hit_intervals or (start - hit_intervals[-1][1]) >= min_hit_separation:
                hit_intervals.append((start, end))
        i += 1
        j += 1
    elif accel_peaks[i] < gyro_peaks[j]:
        i += 1
    else:
        j += 1


plt.figure(figsize=(12, 6))
plt.plot(time, accel_magnitude, label="Accelerometer Magnitude", color='blue')
plt.plot(time, gyro_magnitude, label="Gyroscope Magnitude", color='orange')


for start, end in hit_intervals:
    plt.axvspan(start, end, color='green', alpha=0.3)

plt.xlabel("Time")
plt.ylabel("Magnitude")
plt.legend()
plt.title("Detected Hit Events with Duration and Separation Constraints")
plt.show()


print(f"Total detected hit events: {len(hit_intervals)}")
