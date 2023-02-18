import copy
import numpy as np

from gym_art.quadrotor_multi.scenarios.utils import get_z_value
from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_swarm_vs_swarm(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.goals_1, self.goals_2 = None, None
        self.goal_center_1, self.goal_center_2 = None, None

    def formation_centers(self):
        if self.formation_center is None:
            self.formation_center = np.array([0., 0., 2.])

        # self.envs[0].box = 2.0
        box_size = self.envs[0].box
        dist_low_bound = self.lowest_formation_size
        # Get the 1st goal center
        x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
        # Get z value, and make sure all goals will above the ground
        z = get_z_value(num_agents=self.num_agents, num_agents_per_layer=self.num_agents_per_layer,
                        box_size=box_size, formation=self.formation, formation_size=self.formation_size)

        goal_center_1 = np.array([x, y, z])

        # Get the 2nd goal center
        goal_center_distance = np.random.uniform(low=box_size / 4, high=box_size)

        phi = np.random.uniform(low=-np.pi, high=np.pi)
        theta = np.random.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
        goal_center_2 = goal_center_1 + goal_center_distance * np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        diff_x, diff_y, diff_z = goal_center_2 - goal_center_1
        if self.formation.endswith("horizontal"):
            if abs(diff_z) < dist_low_bound:
                goal_center_2[2] = np.sign(diff_z) * dist_low_bound + goal_center_1[2]
        elif self.formation.endswith("vertical_xz"):
            if abs(diff_y) < dist_low_bound:
                goal_center_2[1] = np.sign(diff_y) * dist_low_bound + goal_center_1[1]
        elif self.formation.endswith("vertical_yz"):
            if abs(diff_x) < dist_low_bound:
                goal_center_2[0] = np.sign(diff_x) * dist_low_bound + goal_center_1[0]

        return goal_center_1, goal_center_2

    def create_formations(self, goal_center_1, goal_center_2):
        self.goals_1 = self.generate_goals(num_agents=self.num_agents // 2, formation_center=goal_center_1,
                                           layer_dist=self.layer_dist)
        self.goals_2 = self.generate_goals(num_agents=self.num_agents - self.num_agents // 2,
                                           formation_center=goal_center_2, layer_dist=self.layer_dist)
        self.goals = np.concatenate([self.goals_1, self.goals_2])

    def update_goals(self):
        tmp_goal_center_1 = copy.deepcopy(self.goal_center_1)
        tmp_goal_center_2 = copy.deepcopy(self.goal_center_2)
        self.goal_center_1 = tmp_goal_center_2
        self.goal_center_2 = tmp_goal_center_1

        self.update_formation_and_relate_param()
        self.create_formations(self.goal_center_1, self.goal_center_2)
        # Shuffle goals
        np.random.shuffle(self.goals_1)
        np.random.shuffle(self.goals_2)
        self.goals = np.concatenate([self.goals_1, self.goals_2])
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

    def step(self, infos, pos):
        tick = self.envs[0].tick
        # Switch every [4, 6] seconds
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()
        return infos

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset the formation size and the goals of swarms
        self.goal_center_1, self.goal_center_2 = self.formation_centers()
        self.create_formations(self.goal_center_1, self.goal_center_2)

        # This is for initialize the pos for obstacles
        self.formation_center = (self.goal_center_1 + self.goal_center_2) / 2

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.create_formations(self.goal_center_1, self.goal_center_2)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]
