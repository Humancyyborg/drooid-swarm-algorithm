import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_static_same_goal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.approch_goal_metric = 1.0

    def step(self):
        # tick = self.envs[0].tick
        #
        # if tick <= int(self.duration_time * self.envs[0].control_freq):
        #     return
        #
        # self.duration_time += self.envs[0].ep_time + 1
        # for i, env in enumerate(self.envs):
        #     env.goal = self.end_point

        return

    def reset(self, obst_map=None, cell_centers=None):
        # Update duration time
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(self.duration_time * self.envs[0].control_freq)

        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        # self.start_point = []
        # for i in range(self.num_agents):
        #     self.start_point.append(self.generate_pos_obst_map())
        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        # self.end_point = self.generate_pos_obst_map(check_surroundings=True)
        self.end_point = self.max_square_area_center()

        # test_point = self.max_square_area()

        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reassign goals
        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = np.array([self.end_point for _ in range(self.num_agents)])

    def max_square_area_center(self):
        """
        Finds the maximum square area of 0 in a 2D matrix and returns the coordinates
        of the center element of the largest square area.
        """
        n, m = self.obstacle_map.shape
        # Initialize a 2D numpy array to store the maximum size of square submatrices
        # that end at each element of the matrix.
        dp = np.zeros((n, m), dtype=int)
        # Initialize the first row and first column of the dp array
        dp[0] = self.obstacle_map[0]
        dp[:, 0] = self.obstacle_map[:, 0]
        # Initialize variables to store the maximum square area and its center coordinates
        max_size = 0
        center_x = 0
        center_y = 0
        # Fill the remaining entries of the dp array
        for i in range(1, n):
            for j in range(1, m):
                if self.obstacle_map[i][j] == 0:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    if dp[i][j] > max_size:
                        max_size = dp[i][j]
                        center_x = i - (max_size - 1) // 2
                        center_y = j - (max_size - 1) // 2
        # Return the center coordinates of the largest square area as a tuple
        index = center_x + (m * center_y)
        pos_x, pos_y = self.cell_centers[index]
        z_list_start = np.random.uniform(low=0.75, high=3.0)
        return np.array([pos_x, pos_y, z_list_start])
