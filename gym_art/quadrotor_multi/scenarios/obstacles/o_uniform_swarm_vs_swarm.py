import copy
import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_o_uniform_swarm_vs_swarm(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)
        self.goals_1, self.goals_2 = None, None
        self.goal_center_1, self.goal_center_2 = None, None
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_time = 0.0
        self.quads_mode = quads_mode
        self.env_shuffle_list = np.arange(len(envs))
        self.cur_start_tick = 0

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.create_formations(self.goal_center_1, self.goal_center_2)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def generate_centers(self):
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        x_1, x_2 = np.random.uniform(low=-1.0 * half_room_length + 1.1, high=half_room_length - 1.1, size=2)
        y_1, y_2 = np.random.uniform(low=-1.0 * half_room_width + 1.1, high=half_room_width - 1.1, size=2)
        z_1, z_2 = np.random.uniform(low=1.0, high=4.0, size=2)

        pos_1 = np.array([x_1, y_1, z_1])
        pos_2 = np.array([x_2, y_2, z_2])

        return pos_1, pos_2

    def create_formations(self, goal_center_1, goal_center_2):
        self.goals_1 = self.generate_goals(num_agents=self.num_agents // 2, formation_center=goal_center_1,
                                           layer_dist=self.layer_dist)
        self.goals_2 = self.generate_goals(num_agents=self.num_agents - self.num_agents // 2,
                                           formation_center=goal_center_2, layer_dist=self.layer_dist)
        # Shuffle goals
        np.random.shuffle(self.goals_1)
        np.random.shuffle(self.goals_2)
        tmp_goals = np.concatenate([self.goals_1, self.goals_2])
        self.goals = copy.deepcopy(tmp_goals)
        for i in range(len(self.envs)):
            self.goals[self.env_shuffle_list[i]] = tmp_goals[i]

    def update_goals(self):
        tmp_goal_center_1 = copy.deepcopy(self.goal_center_1)
        tmp_goal_center_2 = copy.deepcopy(self.goal_center_2)
        self.goal_center_1 = tmp_goal_center_2
        self.goal_center_2 = tmp_goal_center_1

        self.update_formation_and_relate_param()
        self.create_formations(self.goal_center_1, self.goal_center_2)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

    def step(self, infos, pos):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            return infos

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        self.update_goals()
        self.duration_time += self.envs[0].ep_time + 1
        return infos

    def reset(self):
        self.cur_start_tick = 0
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        np.random.shuffle(self.env_shuffle_list)
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset the formation size and the goals of swarms
        self.goal_center_1, self.goal_center_2 = self.generate_centers()
        self.start_point = copy.deepcopy(self.goal_center_1)
        self.end_point = copy.deepcopy(self.goal_center_2)
        self.create_formations(self.goal_center_1, self.goal_center_2)
        self.formation_center = (self.goal_center_1 + self.goal_center_2) / 2
