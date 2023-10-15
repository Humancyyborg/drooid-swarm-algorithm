import plotly.graph_objects as go
import numpy as np
import ast

# Function to read and parse trajectory data from a file
def read_trajectory(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    trajectory = [ast.literal_eval(line.strip()) for line in lines]
    return np.array(trajectory)

# Read the data from the files
file_paths = ["pos_3.txt"]  # Replace with actual file paths
trajectories = {file_path: read_trajectory(file_path) for file_path in file_paths}

# Create the interactive 3D plot
fig = go.Figure()

for file_path, trajectory in trajectories.items():
    fig.add_trace(go.Scatter3d(x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
                               mode='lines+markers',
                               name=f'{file_path.split("/")[-1]}'))

fig.update_layout(scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                  title="Interactive 3D Trajectories")

fig.show()
