import os
import sys
import time

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

num_agents = 8
scenaio_name = 'o_random'
color_list = ['red', 'orange', 'yellow', 'cyan', 'magenta', 'blue', 'purple', 'violet', 'brown', 'gray', 'olive']
obstacle_radius = 0.85 * 0.5
trail_length = 200

experiment_name = 'None'


def load_record_data(file_counter=0):
    str_counter = str(file_counter)
    load_folder_path = os.path.join('data_for_video', experiment_name, scenaio_name, str_counter)
    print('load record data, folder name: ', load_folder_path)

    acc_lists = []
    for i in range(num_agents):
        file_name = os.path.join(load_folder_path, 'acc_list_' + str(i) + '.csv')
        acc_list = pd.read_csv(file_name)
        acc_lists.append(acc_list)

    pv_lists = []
    for i in range(num_agents):
        file_name = os.path.join(load_folder_path, 'pv_list_' + str(i) + '.csv')
        pv_list = pd.read_csv(file_name)
        pv_lists.append(pv_list)

    file_name = os.path.join(load_folder_path, 'obstacle_positions.csv')
    obstacle_positions = pd.read_csv(file_name)
    file_name = os.path.join(load_folder_path, 'goal.csv')
    goal_positions = pd.read_csv(file_name)

    obstacle_pos = obstacle_positions[['X', 'Y']].values
    goal_pos = goal_positions[['X', 'Y']].values

    return acc_lists, pv_lists, obstacle_positions, goal_positions, obstacle_pos, goal_pos


# Animation update function
def update(frame, acc_lists, pv_lists, obstacle_positions, goal_positions, ax, trails, trail_buffers):
    # Clear previous frame
    ax.clear()

    # Set limits and labels
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('Range/m')

    # Plot obstacles
    for _, obstacle in obstacle_positions.iterrows():
        circle = plt.Circle((obstacle['X'], obstacle['Y']), radius=obstacle_radius, color='green')
        ax.add_artist(circle)

    for g_id, goal in goal_positions.iterrows():
        circle = plt.Circle((goal['X'], goal['Y']), radius=0.1, color=color_list[g_id])
        ax.add_artist(circle)

    # Plot drones
    drone_poses = []
    for i in range(num_agents):
        drone_pos = pv_lists[i].iloc[frame, :2]
        ax.plot(drone_pos[0], drone_pos[1], 'o', color=color_list[i], label='Drone' + str(i))
        drone_poses.append(drone_pos)
        # Update trail buffer
        trail_buffers[i].append(drone_pos)
        x_trail, y_trail = zip(*trail_buffers[i])
        ax.plot(x_trail, y_trail, '--', color=color_list[i], lw=1, alpha=0.5)

    # Plot acceleration vectors
    acc_scale = 0.1  # Scale for the acceleration vectors for visibility
    vel_scale = 0.2
    acc_ref_color = color_list[0]  # red
    acc_sbc_color = color_list[5]  # blue
    acc_real_color = color_list[8]  # brown

    vel_color = color_list[1]
    for i in range(num_agents):
        vel = pv_lists[i].iloc[frame, 3:5]
        ax.arrow(drone_poses[i][0], drone_poses[i][1], vel_scale * vel[0], vel_scale * vel[1],
                 head_width=0.1, head_length=0.1, fc=vel_color, ec=vel_color)

    for i in range(num_agents):
        acc_ref = acc_lists[i].iloc[frame, :3]
        ax.arrow(drone_poses[i][0], drone_poses[i][1], acc_scale * acc_ref[0], acc_scale * acc_ref[1],
                 head_width=0.1, head_length=0.1, fc=acc_ref_color, ec=acc_ref_color)

        acc_sbc = acc_lists[i].iloc[frame, 3:6]
        ax.arrow(drone_poses[i][0], drone_poses[i][1], acc_scale * acc_sbc[0], acc_scale * acc_sbc[1],
                 head_width=0.1, head_length=0.1, fc=acc_sbc_color, ec=acc_sbc_color)

        acc_real = acc_lists[i].iloc[frame, 6:9]
        ax.arrow(drone_poses[i][0], drone_poses[i][1], acc_scale * acc_real[0], acc_scale * acc_real[1],
                 head_width=0.1, head_length=0.1, fc=acc_real_color, ec=acc_real_color)


def process_animation(file_counter):
    # Set up the figure and axis for animation
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('Range/m')
    ax.set_ylabel('Range/m')

    start_time = time.time()
    acc_lists, pv_lists, obstacle_positions, goal_positions, obstacle_pos, goal_pos = load_record_data(file_counter)

    # Plot static obstacles
    obstacles, = ax.plot(obstacle_pos[:, 0], obstacle_pos[:, 1], 'o', color='blue', label='Obstacles')
    goals, = ax.plot(goal_pos[:, 0], goal_pos[:, 1], 'o', color='yellow', label='Goals')

    # Initialize two drones with different colors
    drones = []
    for i in range(num_agents):
        drone, = ax.plot([], [], 'o', color=color_list[i], label='Drone' + str(i))
        drones.append(drone)

    # Initialize velocity vectors for drones
    vels = []
    for i in range(8):
        velocity = ax.quiver([], [], [], [], angles='xy', scale_units='xy', scale=1, color=color_list[i])
        vels.append(velocity)

    # Initialize trails for drones
    trails = []
    # Length of the trails
    trail_buffers = [deque(maxlen=trail_length) for _ in range(num_agents)]
    for i in range(num_agents):
        trail = ax.plot([], [], '--', color=color_list[i], lw=1, alpha=0.5)
        trails.append(trail)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create animation
    print('start animation: ' + str(file_counter))
    ani = FuncAnimation(fig, update, frames=range(len(pv_lists[0])),
                        fargs=(acc_lists, pv_lists, obstacle_positions, goal_positions, ax, trails, trail_buffers),
                        interval=100)

    folder_path = os.path.join('videos', scenaio_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    # Save the animation as a video file
    video_path = os.path.join(folder_path, experiment_name, 'drone_animation_' + str(file_counter) + '.mp4')
    print('start saving: ' + str(file_counter))
    ani.save(video_path, writer='ffmpeg', fps=20)

    time_used = time.time() - start_time
    print('use_time: ', time_used)


def main():
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Adjust max_workers based on your system capabilities
        file_counters = range(11, 20)  # Example range, adjust as needed
        futures = [executor.submit(process_animation, file_counter) for file_counter in file_counters]

        # Wait for all tasks to complete (optional, depending on your use case)
        for future in futures:
            future.result()


if __name__ == '__main__':
    sys.exit(main())
