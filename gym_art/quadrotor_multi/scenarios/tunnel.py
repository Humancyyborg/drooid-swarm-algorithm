import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_tunnel(QuadrotorScenario):
    def update_goals(self, formation_center):
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=formation_center,
                                         layer_dist=self.layer_dist)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        # hack to make drones and goals be on opposite sides of the tunnel
        t = self.envs[0].tick
        if t == 1:
            for env in self.envs:
                if abs(env.goal[0]) > abs(env.goal[1]):
                    env.goal[0] = -env.goal[0]
                else:
                    env.goal[1] = -env.goal[1]
        return infos, rewards

    def reset(self):
        # tunnel could be in the x or y direction
        p = np.random.uniform(0, 1)
        if p <= 0.5:
            self.update_room_dims((10, 2, 2))
            formation_center = np.array([-4, 0, 1])
        else:
            self.update_room_dims((2, 10, 2))
            formation_center = np.array([0, -4, 1])
        self.update_goals(formation_center)
