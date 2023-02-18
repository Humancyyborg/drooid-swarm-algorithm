import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_ep_lissajous3D(QuadrotorScenario):
    # Based on https://mathcurve.com/courbes3d.gb/lissajous3d/lissajous3d.shtml
    @staticmethod
    def lissajous3D(tick, a=0.03, b=0.01, c=0.01, n=2, m=2, phi=90, psi=90):
        x = a * np.sin(tick)
        y = b * np.sin(n * tick + phi)
        z = c * np.cos(m * tick + psi)
        return x, y, z

    def step(self, infos, pos):
        control_freq = self.envs[0].control_freq
        tick = self.envs[0].tick / control_freq
        x, y, z = self.lissajous3D(tick)
        goal_x, goal_y, goal_z = self.goals[0]
        x_new, y_new, z_new = x + goal_x, y + goal_y, z + goal_z
        self.goals = np.array([[x_new, y_new, z_new] for _ in range(self.num_agents)])

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos

    def update_formation_size(self, new_formation_size):
        pass

    def reset(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Generate goals
        self.formation_center = np.array([-2.0, 0.0, 2.0])  # prevent drones from crashing into the wall
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=0.0)
