import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_o_uniform_same_goal_spawn(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_time = 0.0
        self.quads_mode = quads_mode

    def update_formation_size(self, new_formation_size):
        pass

    def generate_pos(self):
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        x = np.random.uniform(low=-1.0 * half_room_length + 1.1, high=half_room_length - 1.1)
        y = np.random.uniform(low=-1.0 * half_room_width + 1.1, high=half_room_width - 1.1)

        z = np.random.uniform(low=1.0, high=4.0)

        return np.array([x, y, z])

    def step(self, infos, pos):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            return infos

        self.duration_time += self.envs[0].ep_time + 1
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos

    def reset(self):
        self.start_point = self.generate_pos()
        self.end_point = self.generate_pos()
        self.duration_time = np.random.uniform(low=2.0, high=4.0)
        self.standard_reset(formation_center=self.start_point)