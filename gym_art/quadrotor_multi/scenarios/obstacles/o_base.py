import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_o_base(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_time = 0.0
        self.quads_mode = quads_mode
        self.obstacle_map = None
        self.free_space = []
        self.grid_size = 1.0
        self.cell_centers = [
            [i + (self.grid_size / 2) - self.room_dims[0] / 2, j + (self.grid_size / 2) - self.room_dims[1] / 2] for i
            in
            np.arange(0, self.room_dims[0], self.grid_size) for j in
            np.arange(self.room_dims[1] - self.grid_size, -self.grid_size, -self.grid_size)]

    def update_formation_size(self, new_formation_size):
        pass

    def generate_pos(self):
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        x = np.random.uniform(low=-1.0 * half_room_length + 2.0, high=half_room_length - 2.0)
        y = np.random.uniform(low=-1.0 * half_room_width + 2.0, high=half_room_width - 2.0)

        z = np.random.uniform(low=1.0, high=4.0)

        return np.array([x, y, z])

    def step(self, infos, rewards):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            return infos, rewards

        self.duration_time += self.envs[0].ep_time + 1
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos, rewards

    def reset(self, obst_map=None):
        self.start_point = self.generate_pos()
        self.end_point = self.generate_pos()
        self.duration_time = np.random.uniform(low=2.0, high=4.0)
        self.standard_reset(formation_center=self.start_point)

    def generate_pos_obst_map(self):
        idx = np.random.choice(a=len(self.free_space), replace=True)
        z_list_start = np.random.uniform(low=0.5, high=3.0)
        xy_noise = np.random.uniform(low=-0.5, high=0.5, size=2)

        x, y = self.free_space[idx][0], self.free_space[idx][1]
        index = x + (10 * y)
        pos_x, pos_y = self.cell_centers[index]

        return np.array([pos_x + xy_noise[0], pos_y + xy_noise[1], z_list_start])
