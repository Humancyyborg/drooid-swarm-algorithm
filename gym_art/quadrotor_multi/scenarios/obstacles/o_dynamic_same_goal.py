import numpy as np

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_dynamic_same_goal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_formation_size(self, new_formation_size):
        pass

    def step(self, infos, rewards):
        tick = self.envs[0].tick
        if tick % self.control_step_for_sec == 0 and tick > 0:
            box_size = self.envs[0].box
            self.formation_center = self.generate_pos_obst_map()
            self.formation_center[2] = max(0.25, self.formation_center[2])

            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                             layer_dist=0.0)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return infos, rewards

    def reset(self, obst_map=None):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        self.obstacle_map = obst_map
        if obst_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        self.formation_center = self.generate_pos_obst_map()

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)