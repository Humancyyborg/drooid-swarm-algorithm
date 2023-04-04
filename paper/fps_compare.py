import os

import matplotlib.pyplot as plt
import numpy as np

# Define the data
x = [1, 8, 32, 128]
y_quad_swarm = [48589, 62042, 60241, 38449]
y_pybullet = [21883, 31539, 31457.28, 32522]

# Set the width of each bar and the positions of the x-ticks
bar_width = 0.35
x_pos = np.arange(len(x))

# Create a figure and axes object
fig, ax = plt.subplots()

# Plot the two groups as grouped bars
rects1 = ax.bar(x_pos - bar_width/2, y_pybullet, bar_width, label='gym-pybullet-drones')
rects2 = ax.bar(x_pos + bar_width/2, y_quad_swarm, bar_width, label='QuadSwarm')


# Add labels and legend
ax.set_xlabel('Number of Quadrotors')
ax.set_ylabel('Simulation Samples Per Second (SPS)')
# ax.set_title('Comparison of Quad Swarm and PyBullet')
ax.set_xticks(x_pos)
ax.set_xticklabels(x)
lgd = ax.legend(bbox_to_anchor=(0.02, 0.95, 0.95, 0.17), loc='upper left', ncol=2, mode="expand",
                 prop={'size': 12})
lgd.set_in_layout(True)

# Show the plot
# plt.show()
plt.savefig(os.path.join(os.getcwd(), f'quads_train_setting.pdf'), format='pdf', bbox_inches='tight', pad_inches=0.01)