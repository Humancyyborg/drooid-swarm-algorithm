import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_dynamic_formations(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)
        # if increase_formation_size is True, increase the formation size
        # else, decrease the formation size
        self.increase_formation_size = True
        # low: 0.1m/s, high: 0.3m/s
        self.control_speed = np.random.uniform(low=1.0, high=3.0)

    # change formation sizes on the fly
    def update_goals(self):
        self.goals = self.generate_goals(self.num_agents, self.formation_center, layer_dist=self.layer_dist)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, pos):
        if self.formation_size <= -self.highest_formation_size:
            self.increase_formation_size = True
            self.control_speed = np.random.uniform(low=1.0, high=3.0)
        elif self.formation_size >= self.highest_formation_size:
            self.increase_formation_size = False
            self.control_speed = np.random.uniform(low=1.0, high=3.0)

        if self.increase_formation_size:
            self.formation_size += 0.001 * self.control_speed
        else:
            self.formation_size -= 0.001 * self.control_speed

        self.update_goals()
        return infos

    def reset(self):
        self.increase_formation_size = True if np.random.uniform(low=0.0, high=1.0) < 0.5 else False
        self.control_speed = np.random.uniform(low=1.0, high=3.0)

        # Reset formation, and parameters related to the formation; formation center; goals
        self.standard_reset()

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()
