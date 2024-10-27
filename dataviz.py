import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv('cleaned_data4.csv')


hit_data = df[df['hit'] == True]
non_hit_data = df[df['hit'] == False]


fig = plt.figure(figsize=(20, 10))


ax1 = fig.add_subplot(221, projection='3d')


sc1 = ax1.scatter(
    hit_data['x'], 
    hit_data['y'], 
    hit_data['z'], 
    c=hit_data['accel_magnitude'], 
    cmap='Reds', 
    s=hit_data['accel_magnitude_roll_avg'] * 50, 
    alpha=0.7, 
    label='Hit Accelerometer'
)


sc1_non_hit = ax1.scatter(
    non_hit_data['x'], 
    non_hit_data['y'], 
    non_hit_data['z'], 
    c=non_hit_data['accel_magnitude'], 
    cmap='Blues', 
    s=non_hit_data['accel_magnitude_roll_avg'] * 50, 
    alpha=0.7, 
    label='Not Hit Accelerometer'
)


ax1.set_xlabel('X Axis')
ax1.set_ylabel('Y Axis')
ax1.set_zlabel('Z Axis')
ax1.set_title('Accelerometer Raw Data (x, y, z)')
cbar1 = plt.colorbar(sc1, ax=ax1, label='Accel Magnitude')


ax1.legend(loc='upper right')


ax2 = fig.add_subplot(222, projection='3d')


sc2 = ax2.scatter(
    hit_data['x_roll_avg'], 
    hit_data['y_roll_avg'], 
    hit_data['z_roll_avg'],
    c=hit_data['accel_magnitude'], 
    cmap='Reds', 
    s=hit_data['accel_magnitude_roll_avg'] * 50, 
    alpha=0.7, 
    label='Hit Rolling Avg'
)


sc2_non_hit = ax2.scatter(
    non_hit_data['x_roll_avg'], 
    non_hit_data['y_roll_avg'], 
    non_hit_data['z_roll_avg'],
    c=non_hit_data['accel_magnitude'], 
    cmap='Blues', 
    s=non_hit_data['accel_magnitude_roll_avg'] * 50, 
    alpha=0.7, 
    label='Not Hit Rolling Avg'
)


ax2.set_xlabel('X Rolling Avg')
ax2.set_ylabel('Y Rolling Avg')
ax2.set_zlabel('Z Rolling Avg')
ax2.set_title('Accelerometer Rolling Averages')
cbar2 = plt.colorbar(sc2, ax=ax2, label='Accel Magnitude')


ax2.legend(loc='upper right')


ax3 = fig.add_subplot(223, projection='3d')


sc3 = ax3.scatter(
    hit_data['gx'], 
    hit_data['gy'], 
    hit_data['gz'], 
    c=hit_data['gyro_magnitude'], 
    cmap='Oranges', 
    s=hit_data['gyro_magnitude_roll_avg'] * 50, 
    alpha=0.7, 
    label='Hit Gyroscope'
)


sc3_non_hit = ax3.scatter(
    non_hit_data['gx'], 
    non_hit_data['gy'], 
    non_hit_data['gz'], 
    c=non_hit_data['gyro_magnitude'], 
    cmap='Greens', 
    s=non_hit_data['gyro_magnitude_roll_avg'] * 50, 
    alpha=0.7, 
    label='Not Hit Gyroscope'
)


ax3.set_xlabel('GX Axis')
ax3.set_ylabel('GY Axis')
ax3.set_zlabel('GZ Axis')
ax3.set_title('Gyroscope Raw Data (gx, gy, gz)')
cbar3 = plt.colorbar(sc3, ax=ax3, label='Gyro Magnitude')


ax3.legend(loc='upper right')


ax4 = fig.add_subplot(224, projection='3d')


sc4 = ax4.scatter(
    hit_data['gx_roll_avg'], 
    hit_data['gy_roll_avg'], 
    hit_data['gz_roll_avg'],
    c=hit_data['gyro_magnitude'], 
    cmap='Oranges', 
    s=hit_data['gyro_magnitude_roll_avg'] * 50, 
    alpha=0.7, 
    label='Hit Gyro Rolling Avg'
)


sc4_non_hit = ax4.scatter(
    non_hit_data['gx_roll_avg'], 
    non_hit_data['gy_roll_avg'], 
    non_hit_data['gz_roll_avg'],
    c=non_hit_data['gyro_magnitude'], 
    cmap='Greens', 
    s=non_hit_data['gyro_magnitude_roll_avg'] * 50, 
    alpha=0.7, 
    label='Not Hit Gyro Rolling Avg'
)


ax4.set_xlabel('GX Rolling Avg')
ax4.set_ylabel('GY Rolling Avg')
ax4.set_zlabel('GZ Rolling Avg')
ax4.set_title('Gyroscope Rolling Averages')
cbar4 = plt.colorbar(sc4, ax=ax4, label='Gyro Magnitude')


ax4.legend(loc='upper right')


plt.tight_layout()
plt.show()
