import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv('cleaned_data.csv')

hit_data = df[df['hit'] == True]
non_hit_data = df[df['hit'] == False]


fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')


sc1 = ax.scatter(hit_data['x'], hit_data['y'], hit_data['z'], 
                 c=hit_data['accel_magnitude'], cmap='Reds', s=hit_data['accel_magnitude_roll_avg']*50, 
                 alpha=0.7, label='Hit')


sc2 = ax.scatter(non_hit_data['x'], non_hit_data['y'], non_hit_data['z'], 
                 c=non_hit_data['gyro_magnitude'], cmap='Blues', s=non_hit_data['accel_magnitude_roll_avg']*50, 
                 alpha=0.3, label='Not Hit')


ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Motion Data: Hits vs Not Hits with Magnitude and Rolling Average Insights')


cbar1 = plt.colorbar(sc1, ax=ax, orientation='vertical', pad=0.1, label='Accel Magnitude (Hits)')
cbar2 = plt.colorbar(sc2, ax=ax, orientation='vertical', pad=0.05, label='Gyro Magnitude (Not Hits)')


ax.legend(loc='upper right')

plt.show()
