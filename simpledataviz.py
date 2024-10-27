import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_3d_distance(df):
    df['distance'] = np.sqrt(
        (df['x'].diff())**2 + 
        (df['y'].diff())**2 + 
        (df['z'].diff())**2
    )
    return df


df = pd.read_csv("cleaned_data4.csv")


hit_data = df[df['hit'] == True]
non_hit_data = df[df['hit'] == False]


hit_data = calculate_3d_distance(hit_data)


total_distance = hit_data['distance'].sum()
print(f"Total distance traveled during hits: {total_distance:.2f} units")


hit_data['time_normalized'] = (hit_data['time'] - hit_data['time'].min()) / (hit_data['time'].max() - hit_data['time'].min())


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


sc = ax.scatter(hit_data['gx_roll_avg'], hit_data['gy_roll_avg'], hit_data['gz_roll_avg'], 
                c=hit_data['time_normalized'], cmap='coolwarm', s=100, label='Hit')


ax.scatter(non_hit_data['gx_roll_avg'], non_hit_data['gy_roll_avg'], non_hit_data['gz_roll_avg'], c='b', marker='^', label='Not Hit')


ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Motion Data: Hit vs Not Hit (Color-Coded by Time)')


colorbar = plt.colorbar(sc, ax=ax)
colorbar.set_label('Time (Normalized)')


ax.legend()


plt.show()
