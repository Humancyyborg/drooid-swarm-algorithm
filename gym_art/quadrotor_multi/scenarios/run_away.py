import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_run_away(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)

    def update_goals(self):
        self.goals = self.generate_goals(self.num_agents, self.formation_center, layer_dist=self.layer_dist)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, pos):
        tick = self.envs[0].tick
        control_step_for_sec = int(1.0 * self.envs[0].control_freq)

        if tick % control_step_for_sec == 0 and tick > 0:
            g_index = np.random.randint(low=1, high=self.num_agents, size=2)
            self.goals[0] = self.goals[g_index[0]]
            self.goals[1] = self.goals[g_index[1]]
            self.envs[0].goal = self.goals[0]
            self.envs[1].goal = self.goals[1]

        return infos

    def reset(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()
        # Reset formation center
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()
