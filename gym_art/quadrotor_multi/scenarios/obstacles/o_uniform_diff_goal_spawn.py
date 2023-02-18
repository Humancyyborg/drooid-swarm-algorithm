import numpy as np

from gym_art.quadrotor_multi.scenarios.obstacles.o_uniform_same_goal_spawn import Scenario_o_uniform_same_goal_spawn


class Scenario_o_uniform_diff_goal_spawn(Scenario_o_uniform_same_goal_spawn):
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
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                             layer_dist=self.layer_dist)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def update_goals(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset goals
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        np.random.shuffle(self.goals)

    def step(self, infos, pos):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            return infos

        self.duration_time += self.envs[0].ep_time + 1
        self.update_goals()
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]
        return infos
