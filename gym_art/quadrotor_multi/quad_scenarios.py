import numpy as np
import bezier
import copy

from gym_art.quadrotor_multi.quad_scenarios_utils import QUADS_PARAMS_DICT, update_formation_and_max_agent_per_layer, \
    update_layer_dist, get_formation_range, get_goal_by_formation, get_z_value, QUADS_MODE_LIST, \
    QUADS_MODE_LIST_OBSTACLES, QUADS_MODE_GOAL_CENTERS, QUADS_MODE_OBST_INFO_LIST, get_pos_diff_decay_rate

from gym_art.quadrotor_multi.quad_utils import generate_points, get_grid_dim_number


def create_scenario(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
    cls = eval('Scenario_' + quads_mode)
    scenario = cls(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
    return scenario


class QuadrotorScenario:
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        self.quads_mode = quads_mode
        self.envs = envs
        self.num_agents = num_agents
        self.room_dims = room_dims
        self.set_room_dims = room_dims_callback  # usage example: self.set_room_dims((10, 10, 10))
        self.rew_coeff = rew_coeff
        self.goals = None
        self.one_pass_per_episode = one_pass_per_episode
        self.pos_decay_rate = 0.999
        self.cur_start_tick = 0

        #  Set formation, num_agents_per_layer, lowest_formation_size, highest_formation_size, formation_size,
        #  layer_dist, formation_center
        #  Note: num_agents_per_layer for scalability, the maximum number of agent per layer
        self.formation = quads_formation
        self.num_agents_per_layer = 8
        quad_arm = self.envs[0].dynamics.arm
        self.lowest_formation_size, self.highest_formation_size = 8 * quad_arm, 16 * quad_arm
        self.formation_size = quads_formation_size
        self.layer_dist = self.lowest_formation_size
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Reset episode time
        # if self.quads_mode != 'mix':
        #     ep_time = QUADS_PARAMS_DICT[quads_mode][2]
        # else:
        #     ep_time = QUADS_PARAMS_DICT['swap_goals'][2]
        #
        # for env in self.envs:
        #     env.reset_ep_len(ep_time=ep_time)

        # Aux variables for scenario: pursuit evasion
        self.interp = None

        # Aux obstacle
        self.obst_num_in_room = 0

    def name(self):
        """
        :return: scenario name
        """
        return self.__class__.__name__

    def generate_goals(self, num_agents, formation_center=None, layer_dist=0.0):
        if formation_center is None:
            formation_center = np.array([0., 0., 2.])

        if self.formation.startswith("circle"):
            if num_agents <= self.num_agents_per_layer:
                real_num_per_layer = [num_agents]
            else:
                whole_layer_num = num_agents // self.num_agents_per_layer
                real_num_per_layer = [self.num_agents_per_layer for _ in range(whole_layer_num)]
                rest_num = num_agents % self.num_agents_per_layer
                if rest_num > 0:
                    real_num_per_layer.append(rest_num)

            pi = np.pi
            goals = []
            for i in range(num_agents):
                cur_layer_num_agents = real_num_per_layer[i // self.num_agents_per_layer]
                degree = 2 * pi * (i % cur_layer_num_agents) / cur_layer_num_agents
                pos_0 = self.formation_size * np.cos(degree)
                pos_1 = self.formation_size * np.sin(degree)
                goal = get_goal_by_formation(formation=self.formation, pos_0=pos_0, pos_1=pos_1, layer_pos=(i//self.num_agents_per_layer) * layer_dist)
                goals.append(goal)

            goals = np.array(goals)
            goals += formation_center
        elif self.formation == "sphere":
            if num_agents < 3:
                goals = np.zeros((num_agents, 3)) + formation_center
            else:
                goals = self.formation_size * np.array(generate_points(num_agents)) + formation_center
        elif self.formation.startswith("grid"):
            if num_agents <= self.num_agents_per_layer:
                dim_1, dim_2 = get_grid_dim_number(num_agents)
                dim_size_each_layer = [[dim_1, dim_2]]
            else:
                # whole layer
                whole_layer_num = num_agents // self.num_agents_per_layer
                max_dim_1, max_dim_2 = get_grid_dim_number(self.num_agents_per_layer)
                dim_size_each_layer = [[max_dim_1, max_dim_2] for _ in range(whole_layer_num)]

                # deal with the rest of the drones
                rest_num = num_agents % self.num_agents_per_layer
                if rest_num > 0:
                    dim_1, dim_2 = get_grid_dim_number(rest_num)
                    dim_size_each_layer.append([dim_1, dim_2])

            goals = []
            for i in range(num_agents):
                dim_1, dim_2 = dim_size_each_layer[i//self.num_agents_per_layer]
                pos_0 = self.formation_size * (i % dim_2)
                pos_1 = self.formation_size * (int(i / dim_2) % dim_1)
                goal = get_goal_by_formation(formation=self.formation, pos_0=pos_0, pos_1=pos_1, layer_pos=(i//self.num_agents_per_layer) * layer_dist)
                goals.append(goal)

            mean_pos = np.mean(goals, axis=0)
            goals = goals - mean_pos + formation_center
        elif self.formation.startswith("cube"):
            dim_size = np.power(num_agents, 1.0 / 3)
            floor_dim_size = int(dim_size)
            goals = []
            for i in range(num_agents):
                pos_0 = self.formation_size * (int(i / floor_dim_size) % floor_dim_size)
                pos_1 = self.formation_size * (i % floor_dim_size)
                goal = np.array([formation_center[2] + self.formation_size * (i // np.square(floor_dim_size)), pos_0, pos_1])
                goals.append(goal)

            mean_pos = np.mean(goals, axis=0)
            goals = goals - mean_pos + formation_center
        else:
            raise NotImplementedError("Unknown formation")

        # need to update self.envs when update room size / room_dims
        room_box = copy.deepcopy(self.envs[0].room_box)
        room_box[0] += 0.25
        room_box[1] -= 0.25
        goals = np.clip(goals, a_min=room_box[0], a_max=room_box[1])

        return goals

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def update_formation_and_relate_param(self):
        # Reset formation, num_agents_per_layer, lowest_formation_size, highest_formation_size, formation_size, layer_dist
        self.formation, self.num_agents_per_layer = update_formation_and_max_agent_per_layer(mode=self.quads_mode)
        # QUADS_PARAMS_DICT:
        # Key: quads_mode; Value: 0. formation, 1: [formation_low_size, formation_high_size], 2: episode_time
        lowest_dist, highest_dist = QUADS_PARAMS_DICT[self.quads_mode][1]
        self.lowest_formation_size, self.highest_formation_size = \
            get_formation_range(mode=self.quads_mode, formation=self.formation, num_agents=self.num_agents,
                                low=lowest_dist, high=highest_dist, num_agents_per_layer=self.num_agents_per_layer)

        self.formation_size = np.random.uniform(low=self.lowest_formation_size, high=self.highest_formation_size)
        self.layer_dist = update_layer_dist(low=self.lowest_formation_size, high=self.highest_formation_size)

    def step(self, infos, rewards, pos):
        raise NotImplementedError("Implemented in a specific scenario")

    def reset(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset formation center
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def standard_reset(self, formation_center=None):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset formation center
        if formation_center is None:
            self.formation_center = np.array([0.0, 0.0, 2.0])
        else:
            self.formation_center = formation_center

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)


class Scenario_o_test(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.duration_time = 1.0
        self.obstacle_pos = np.array([0.0, 0.0, 2.0])

    def update_formation_size(self, new_formation_size):
        pass

    def set_end_point(self):
        self.start_point = np.copy(self.end_point)
        shift_z = np.random.uniform(low=-0.2, high=0.2)
        self.end_point = np.array([self.obstacle_pos[0], self.obstacle_pos[1], self.start_point[2] + shift_z])
        self.duration_time += 1.0

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        self.obstacle_pos = infos[0]['obstacles'][0].pos

        if tick < int(self.duration_time * self.envs[0].control_freq):
            return infos, rewards

        self.set_end_point()
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos, rewards

    def reset(self):
        self.start_point = np.array([0.0, 0.0, 5.5])
        self.end_point = np.array([0.0, 0.0, 5.5])
        self.duration_time = 1.0
        self.standard_reset(formation_center=self.start_point)


class Scenario_o_test_stack(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.duration_time = 1.0
        self.obstacle_pos = np.array([0.0, 0.0, 2.0])
        self.spawn_flag = -1

    def update_formation_size(self, new_formation_size):
        pass

    def set_end_point(self):
        self.start_point = np.copy(self.end_point)
        shift_z = np.random.uniform(low=-0.2, high=0.2)
        self.end_point = np.array([self.obstacle_pos[0], self.obstacle_pos[1], self.start_point[2] + shift_z])
        self.duration_time += 1.0

    def stack_goals(self):
        x, y, z = self.start_point
        z_shift = 0.0
        for i in range(len(self.envs)):
            z_shift += 0.5
            self.goals[i] = np.array([x, y, z + z_shift])

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick

        if tick < int(self.duration_time * self.envs[0].control_freq):
            return infos, rewards

        # self.set_end_point()
        # self.stack_goals()
        # self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos, rewards

    def reset(self):
        self.spawn_flag = -1
        self.start_point = np.array([0.0, 0.0, 1.0])
        self.end_point = np.array([0.0, 0.0, 1.0])
        self.duration_time = 100.0
        self.standard_reset(formation_center=self.start_point)
        self.stack_goals()

class Scenario_static_same_goal(QuadrotorScenario):
    def update_formation_size(self, new_formation_size):
        pass

    def step(self, infos, rewards, pos):
        return infos, rewards


class Scenario_static_diff_goal(QuadrotorScenario):
    def step(self, infos, rewards, pos):
        return infos, rewards


class Scenario_dynamic_same_goal(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_formation_size(self, new_formation_size):
        pass

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        if tick % self.control_step_for_sec == 0 and tick > 0:
            box_size = self.envs[0].box
            x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))
            z = np.random.uniform(low=-0.5 * box_size, high=0.5 * box_size) + 2.0
            z = max(0.25, z)
            self.formation_center = np.array([x, y, z])
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=0.0)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return infos, rewards

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation, and parameters related to the formation; formation center; goals
        self.standard_reset()


class Scenario_dynamic_diff_goal(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset goals
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        if tick % self.control_step_for_sec == 0 and tick > 0:
            box_size = self.envs[0].box
            x, y = np.random.uniform(low=-box_size, high=box_size, size=(2,))

            # Get z value, and make sure all goals will above the ground
            z = get_z_value(num_agents=self.num_agents, num_agents_per_layer=self.num_agents_per_layer,
                            box_size=box_size, formation=self.formation, formation_size=self.formation_size)

            self.formation_center = np.array([x, y, z])
            self.update_goals()

            # Update goals to envs
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return infos, rewards

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation, and parameters related to the formation; formation center; goals
        self.standard_reset()


class Scenario_ep_lissajous3D(QuadrotorScenario):
    # Based on https://mathcurve.com/courbes3d.gb/lissajous3d/lissajous3d.shtml
    @staticmethod
    def lissajous3D(tick, a=0.03, b=0.01, c=0.01, n=2, m=2, phi=90, psi=90):
        x = a * np.sin(tick)
        y = b * np.sin(n * tick + phi)
        z = c * np.cos(m * tick + psi)
        return x, y, z

    def step(self, infos, rewards, pos):
        control_freq = self.envs[0].control_freq
        tick = self.envs[0].tick / control_freq
        x, y, z = self.lissajous3D(tick)
        goal_x, goal_y, goal_z = self.goals[0]
        x_new, y_new, z_new = x + goal_x, y + goal_y, z + goal_z
        self.goals = np.array([[x_new, y_new, z_new] for _ in range(self.num_agents)])

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos, rewards

    def update_formation_size(self, new_formation_size):
        pass

    def reset(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Generate goals
        self.formation_center = np.array([-2.0, 0.0, 2.0])  # prevent drones from crashing into the wall
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=0.0)


class Scenario_ep_rand_bezier(QuadrotorScenario):
    def step(self, infos, rewards, pos):
        # randomly sample new goal pos in free space and have the goal move there following a bezier curve
        tick = self.envs[0].tick
        control_freq = self.envs[0].control_freq
        num_secs = 5
        control_steps = int(num_secs * control_freq)
        t = tick % control_steps
        room_dims = np.array(self.room_dims) - self.formation_size
        # min and max distance the goal can spawn away from its current location. 30 = empirical upper bound on
        # velocity that the drones can handle.
        max_dist = min(30, max(room_dims))
        min_dist = max_dist / 2
        if tick % control_steps == 0 or tick == 1:
            # sample a new goal pos that's within the room boundaries and satisfies the distance constraint
            new_goal_found = False
            while not new_goal_found:
                low, high = np.array([-room_dims[0] / 2, -room_dims[1] / 2, 0]), np.array(
                    [room_dims[0] / 2, room_dims[1] / 2, room_dims[2]])
                # need an intermediate point for a deg=2 curve
                new_pos = np.random.uniform(low=-high, high=high, size=(2, 3)).reshape(3, 2)
                # add some velocity randomization = random magnitude * unit direction
                new_pos = new_pos * np.random.randint(min_dist, max_dist + 1) / np.linalg.norm(new_pos, axis=0)
                new_pos = self.goals[0].reshape(3, 1) + new_pos
                lower_bound = np.expand_dims(low, axis=1)
                upper_bound = np.expand_dims(high, axis=1)
                new_goal_found = (new_pos > lower_bound + 0.5).all() and (
                        new_pos < upper_bound - 0.5).all()  # check bounds that are slightly smaller than the room dims
            nodes = np.concatenate((self.goals[0].reshape(3, 1), new_pos), axis=1)
            nodes = np.asfortranarray(nodes)
            pts = np.linspace(0, 1, control_steps)
            curve = bezier.Curve(nodes, degree=2)
            self.interp = curve.evaluate_multi(pts)
            # self.interp = np.clip(self.interp, a_min=np.array([0,0,0.2]).reshape(3,1), a_max=high.reshape(3,1)) # want goal clipping to be slightly above the floor
        if tick % control_steps != 0 and tick > 1:
            self.goals = np.array([self.interp[:, t] for _ in range(self.num_agents)])

            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return infos, rewards

    def update_formation_size(self, new_formation_size):
        pass


class Scenario_swap_goals(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        # teleport every [4.0, 6.0] secs
        duration_time = 5.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        np.random.shuffle(self.goals)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        # Switch every [4, 6] seconds
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()

        return infos, rewards

    def reset(self):
        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation, and parameters related to the formation; formation center; goals
        self.standard_reset()


class Scenario_dynamic_formations(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
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

    def step(self, infos, rewards, pos):
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
        return infos, rewards

    def reset(self):
        self.increase_formation_size = True if np.random.uniform(low=0.0, high=1.0) < 0.5 else False
        self.control_speed = np.random.uniform(low=1.0, high=3.0)

        # Reset formation, and parameters related to the formation; formation center; goals
        self.standard_reset()

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()


class Scenario_swarm_vs_swarm(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
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
        goal_center_distance = np.random.uniform(low=box_size/4, high=box_size)

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
        self.goals_1 = self.generate_goals(num_agents=self.num_agents // 2, formation_center=goal_center_1, layer_dist=self.layer_dist)
        self.goals_2 = self.generate_goals(num_agents=self.num_agents - self.num_agents // 2, formation_center=goal_center_2, layer_dist=self.layer_dist)
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

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        # Switch every [4, 6] seconds
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()
        return infos, rewards

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


class Scenario_tunnel(QuadrotorScenario):
    def update_goals(self, formation_center):
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=formation_center, layer_dist=self.layer_dist)
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


class Scenario_run_away(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)

    def update_goals(self):
        self.goals = self.generate_goals(self.num_agents, self.formation_center, layer_dist=self.layer_dist)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        control_step_for_sec = int(1.0 * self.envs[0].control_freq)

        if tick % control_step_for_sec == 0 and tick > 0:
            g_index = np.random.randint(low=1, high=self.num_agents, size=2)
            self.goals[0] = self.goals[g_index[0]]
            self.goals[1] = self.goals[g_index[1]]
            self.envs[0].goal = self.goals[0]
            self.envs[1].goal = self.goals[1]

        return infos, rewards

    def reset(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()
        # Reset formation center
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()


class Scenario_through_hole(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        # teleport every [4.0, 6.0] secs
        self.start_point = np.array([0.0, -3.0, 1.5])
        self.end_point = np.array([0.0, 3.0, 1.5])
        self.duration_time = 0.0
        self.count = 0

    def update_formation_size(self, new_formation_size):
        pass

    def set_end_point(self):
        self.start_point = np.copy(self.end_point)
        end_x = np.random.uniform(low=-0.5, high=0.5)
        if self.start_point[1] < 0.0:
            end_y = np.random.uniform(low=1.0, high=4.0)
        else:
            end_y = np.random.uniform(low=-4.0, high=-1.0)

        end_z = np.random.uniform(low=0.5, high=2.5)
        self.end_point = np.array([end_x, end_y, end_z])
        self.duration_time += np.random.uniform(low=4.0, high=6.0)

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick

        if self.count > 1:
            return infos, rewards

        if tick < int(self.duration_time * self.envs[0].control_freq):
            return infos, rewards

        self.count += 1
        self.set_end_point()
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos, rewards

    def reset(self):
        self.count = 0
        # Reset formation, and parameters related to the formation; formation center; goals
        x = np.random.uniform(low=-0.5, high=0.5)
        y_flag = np.random.randint(2)
        if y_flag == 0:
            y = np.random.uniform(low=2.75, high=3.0)
        else:
            y = np.random.uniform(low=-3.0, high=-2.75)

        z = np.random.uniform(low=0.5, high=2.5)
        self.start_point = np.array([x, y, z])
        self.set_end_point()
        self.duration_time = 0.0
        self.standard_reset(formation_center=self.start_point)


class Scenario_through_random_obstacles(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        # teleport every [4.0, 6.0] secs
        self.start_point = np.array([0.0, -3.0, 1.5])
        self.end_point = np.array([0.0, 3.0, 1.5])
        self.duration_time = 0.0
        self.count = 0

    def update_formation_size(self, new_formation_size):
        pass

    def set_end_point(self):
        self.start_point = np.copy(self.end_point)
        end_x = np.random.uniform(low=-0.5, high=0.5)
        if self.start_point[1] < 0.0:
            end_y = np.random.uniform(low=1.0, high=4.0)
        else:
            end_y = np.random.uniform(low=-4.0, high=-1.0)

        end_z = np.random.uniform(low=0.5, high=2.5)
        self.end_point = np.array([end_x, end_y, end_z])
        self.duration_time += np.random.uniform(low=4.0, high=6.0)

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick

        if self.count > 1:
            return infos, rewards

        if tick < int(self.duration_time * self.envs[0].control_freq):
            return infos, rewards

        self.count += 1
        self.set_end_point()
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point,
                                         layer_dist=0.0)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return infos, rewards

    def reset(self):
        self.count = 0
        # Reset formation, and parameters related to the formation; formation center; goals
        x = np.random.uniform(low=-0.5, high=0.5)
        y_flag = np.random.randint(2)
        if y_flag == 0:
            y = np.random.uniform(low=3.6, high=3.9) # drones spawn [2.6, 4.9]
        else:
            y = np.random.uniform(low=-3.9, high=-3.6)

        z = np.random.uniform(low=0.5, high=2.5)
        self.start_point = np.array([x, y, z])
        self.set_end_point()
        self.duration_time = 0.0
        self.standard_reset(formation_center=self.start_point)


class Scenario_o_dynamic_same_goal(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_time = 0.0
        if one_pass_per_episode:
            self.per_pass_time = [self.envs[0].ep_time + 1, self.envs[0].ep_time + 2]
        else:
            self.per_pass_time = [10.0, 15.0]

        self.init_flag = 0
        self.spawn_flag = 0  # used for init spawn, check quadrotor_single.py
        self.explore_epsilon = 0.1
        self.quads_mode = quads_mode
        self.cur_start_tick = 0

    def update_formation_size(self, new_formation_size):
        pass

    def generate_pos(self, shift_small=1.25, shift_big=2.0, shift_collide=2.5):
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        if self.init_flag == 0:
            x = np.random.uniform(low=-1.0 * half_room_length + shift_collide, high=half_room_length - shift_collide)
            y = np.random.uniform(low=half_room_width - shift_big, high=half_room_width - shift_small)
        elif self.init_flag == 1:
            x = np.random.uniform(low=half_room_length - shift_big, high=half_room_length - shift_small)
            y = np.random.uniform(low=-1.0 * half_room_width + shift_collide, high=half_room_width - shift_collide)
        elif self.init_flag == 2:
            x = np.random.uniform(low=-1.0 * half_room_length + shift_collide, high=half_room_length - shift_collide)
            y = np.random.uniform(low=-1.0 * half_room_width + shift_small, high=-1.0 * half_room_width + shift_big)
        else:
            x = np.random.uniform(low=-1.0 * half_room_length + shift_small, high=-1.0 * half_room_length + shift_big)
            y = np.random.uniform(low=-1.0 * half_room_width + shift_collide, high=half_room_width - shift_collide)

        z = np.random.uniform(low=2.0, high=3.0)
        return np.array([x, y, z])

    def set_end_point(self, info):
        explore_prob = np.random.uniform(low=0.0, high=1.0)

        if 'num_obst_in_room' in info:
            num_obst_in_room = info['num_obst_in_room']
            max_obst_num = info['max_obst_num']
            obst_change_step = info['obst_change_step']

            if 0 < num_obst_in_room <= max_obst_num:
                area_length = obst_change_step * num_obst_in_room
                area_shift_x, area_shift_y = np.random.uniform(low=-0.25 * area_length, high=0.25 * area_length, size=2)

                area_center = np.array([area_shift_x, area_shift_y])
                direct_dir = area_center - self.start_point[:2]
                direct_mag = np.linalg.norm(direct_dir)
                direct_mag = max(direct_mag, 1e-6)
                direct_norm = direct_dir / np.linalg.norm(direct_mag)

                end_length = np.random.uniform(low=self.room_dims[0] / 2 - 3, high=self.room_dims[0] / 2 - 1)
                end_length = np.clip(end_length, a_min=1.0, a_max=self.room_dims[0] / 2)
                e_x, e_y = area_center + direct_norm * end_length
                e_x = np.clip(e_x, a_min=-self.room_dims[0] / 2 + 1, a_max=self.room_dims[0] / 2 - 1)
                e_y = np.clip(e_y, a_min=-self.room_dims[1] / 2 + 1, a_max=self.room_dims[1] / 2 - 1)

                e_z = np.random.uniform(low=2.0, high=3.0)
                self.end_point = np.array([e_x, e_y, e_z])
                self.start_point = copy.deepcopy(self.end_point)

                return

        if explore_prob < self.explore_epsilon:
            flag = np.random.randint(2)
            if flag == 0:
                self.init_flag = (self.init_flag + 1) % 4
            else:
                self.init_flag = (self.init_flag + 3) % 4
        else:
            self.init_flag = (self.init_flag + 2) % 4

        self.end_point = self.generate_pos(shift_small=1.0, shift_big=2.0, shift_collide=1.0)

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate

            return infos, rewards

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        self.set_end_point(info=infos[0])
        self.duration_time += np.random.uniform(low=self.per_pass_time[0], high=self.per_pass_time[1])
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]
            env.pos_decay_rate = self.pos_decay_rate

        return infos, rewards

    def reset(self):
        self.cur_start_tick = 0
        self.explore_epsilon = np.random.uniform(low=0.1, high=0.2)
        self.init_flag = np.random.randint(4)
        self.spawn_flag = self.init_flag
        self.start_point = self.generate_pos(shift_small=1.25, shift_big=2.0, shift_collide=2.5)
        self.end_point = copy.deepcopy(self.start_point)
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.standard_reset(formation_center=self.start_point)


class Scenario_o_dynamic_diff_goal(Scenario_o_dynamic_same_goal):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_time = 0.0  # used for init spawn
        if one_pass_per_episode:
            self.per_pass_time = [self.envs[0].ep_time + 1, self.envs[0].ep_time + 2]
        else:
            self.per_pass_time = [10.0, 15.0]
        self.init_flag = 0
        self.spawn_flag = 0  # used for init spawn, check quadrotor_single.py
        self.explore_epsilon = 0.1
        self.quads_mode = quads_mode
        self.cur_start_tick = 0

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def update_goals(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset goals
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        np.random.shuffle(self.goals)

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate

            return infos, rewards

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        self.set_end_point(info=infos[0])
        self.duration_time += np.random.uniform(low=self.per_pass_time[0], high=self.per_pass_time[1])
        self.update_goals()
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]
            env.pos_decay_rate = self.pos_decay_rate

        return infos, rewards


class Scenario_o_swarm_vs_swarm(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.goals_1, self.goals_2 = None, None
        self.goal_center_1, self.goal_center_2 = None, None
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.spawn_flag = 0  # used for init spawn, check quadrotor_single.py
        self.duration_time = 0.0
        if one_pass_per_episode:
            self.per_pass_time = [self.envs[0].ep_time + 1, self.envs[0].ep_time + 2]
        else:
            self.per_pass_time = [10.0, 15.0]
        self.quads_mode = quads_mode
        self.env_shuffle_list = np.arange(len(envs))
        self.cur_start_tick = 0

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.create_formations(self.goal_center_1, self.goal_center_2)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def generate_centers(self, shift_small=1.25, shift_big=2.0, shift_collide=2.5):
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        if self.spawn_flag == 0 or self.spawn_flag == 2:
            x_1 = np.random.uniform(low=-1.0 * half_room_length + shift_collide, high=half_room_length - shift_collide)
            y_1 = np.random.uniform(low=half_room_width - shift_big, high=half_room_width - shift_small)

            x_2 = np.random.uniform(low=-1.0 * half_room_length + shift_collide, high=half_room_length - shift_collide)
            y_2 = np.random.uniform(low=-1.0 * half_room_width + shift_small, high=-1.0 * half_room_width + shift_big)
        else:
            x_1 = np.random.uniform(low=half_room_length - shift_big, high=half_room_length - shift_small)
            y_1 = np.random.uniform(low=-1.0 * half_room_width + shift_collide, high=half_room_width - shift_collide)

            x_2 = np.random.uniform(low=-1.0 * half_room_length + shift_small, high=-1.0 * half_room_length + shift_big)
            y_2 = np.random.uniform(low=-1.0 * half_room_width + shift_collide, high=half_room_width - shift_collide)

        z_1, z_2 = np.random.uniform(low=2.0, high=3.0, size=2)

        pos_1 = np.array([x_1, y_1, z_1])
        pos_2 = np.array([x_2, y_2, z_2])
        if self.spawn_flag == 0 or self.spawn_flag == 1:
            return pos_1, pos_2
        else:
            return pos_2, pos_1

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
            env.pos_decay_rate = self.pos_decay_rate

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate

            return infos, rewards

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        self.update_goals()
        self.duration_time += np.random.uniform(low=self.per_pass_time[0], high=self.per_pass_time[1])
        return infos, rewards

    def reset(self):
        self.cur_start_tick = 0
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.spawn_flag = np.random.randint(4)
        np.random.shuffle(self.env_shuffle_list)
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset the formation size and the goals of swarms
        self.goal_center_1, self.goal_center_2 = self.generate_centers()
        self.start_point = copy.deepcopy(self.goal_center_1)
        self.end_point = copy.deepcopy(self.goal_center_2)
        self.create_formations(self.goal_center_1, self.goal_center_2)
        self.formation_center = (self.goal_center_1 + self.goal_center_2) / 2


class Scenario_o_dynamic_formations(Scenario_o_dynamic_diff_goal):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        # if increase_formation_size is True, increase the formation size
        # else, decrease the formation size
        self.increase_formation_size = True
        # low: 0.1m/s, high: 0.2m/s
        self.control_speed = 1.0
        self.low_speed = 1.0
        self.high_speed = 2.0

        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.duration_time = 0.0  # used for init spawn
        self.init_flag = 0
        self.spawn_flag = 0  # used for init spawn, check quadrotor_single.py
        self.quads_mode = quads_mode
        self.cur_start_tick = 0

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.update_goals()

    def set_formation_center(self):
        x_flag = np.random.randint(2)
        if x_flag == 0:
            x = np.random.uniform(low=0.01, high=2.0)
        else:
            x = np.random.uniform(low=-2.0, high=-0.01)

        y_flag = np.random.randint(2)
        if y_flag == 0:
            y = np.random.uniform(low=0.01, high=2.0)
        else:
            y = np.random.uniform(low=-2.0, high=-0.01)

        z = np.random.uniform(low=2.5, high=3.5)
        self.formation_center = np.array([x, y, z])

    def update_goals(self):
        self.goals = self.generate_goals(self.num_agents, self.formation_center, layer_dist=self.layer_dist)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate
            return infos, rewards

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        if self.formation_size <= -self.highest_formation_size:
            self.increase_formation_size = True
            self.control_speed = np.random.uniform(low=self.low_speed, high=self.high_speed)

        elif self.formation_size >= self.highest_formation_size:
            self.increase_formation_size = False
            self.control_speed = np.random.uniform(low=self.low_speed, high=self.high_speed)

        if self.increase_formation_size:
            self.formation_size += 0.001 * self.control_speed
        else:
            self.formation_size -= 0.001 * self.control_speed

        self.update_goals()
        return infos, rewards

    def reset(self):
        self.cur_start_tick = 0
        self.increase_formation_size = True if np.random.uniform(low=0.0, high=1.0) < 0.5 else False
        self.control_speed = np.random.uniform(low=self.low_speed, high=self.high_speed)
        self.set_formation_center()

        self.init_flag = np.random.randint(4)
        self.spawn_flag = self.init_flag
        self.start_point = self.generate_pos(shift_small=1.25, shift_big=2.0, shift_collide=2.5)
        self.end_point = copy.deepcopy(self.start_point)
        self.duration_time = np.random.uniform(low=4.0, high=6.0)

        # Reset formation and related parameters
        self.update_formation_and_relate_param()
        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.start_point,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)


class Scenario_o_ep_lissajous3D(QuadrotorScenario):
    # Based on https://mathcurve.com/courbes3d.gb/lissajous3d/lissajous3d.shtml
    @staticmethod
    def lissajous3D(tick, a=0.03, b=0.01, c=0.01, n=2, m=2, phi=90, psi=90):
        x = a * np.sin(tick)
        y = b * np.sin(n * tick + phi)
        z = c * np.sin(m * tick + psi)
        return x, y, z

    def step(self, infos, rewards, pos):
        control_freq = self.envs[0].control_freq
        tick = self.envs[0].tick / control_freq
        x, y, z = self.lissajous3D(tick)
        goal_x, goal_y, goal_z = self.goals[0]
        x_new, y_new, z_new = x + goal_x, y + goal_y, z + goal_z
        self.goals = np.array([[x_new, y_new, z_new] for _ in range(self.num_agents)])

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

            # Set env decay rate
            env.pos_decay_rate = 1.0

        return infos, rewards

    def update_formation_size(self, new_formation_size):
        pass

    def reset(self):
        self.cur_start_tick = 0
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Generate goals
        x = np.random.uniform(low=-3.0, high=-2.0)
        y = np.random.uniform(low=-2.0, high=2.0)
        z = np.random.uniform(low=1.5, high=2.5)
        self.formation_center = np.array([x, y, z])  # prevent drones from crashing into the wall
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=0.0)


class Scenario_o_dynamic_roller(Scenario_o_dynamic_diff_goal):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.duration_time = 0.0
        self.update_direction = 0  # [0, 1, 2] means update in [x, y, z] direction
        self.direction_flag = 0  # [0, 1] means update in [+, -] direction
        self.init_flag = 0
        self.spawn_flag = 0
        self.cur_start_tick = 0

    def update_goals(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset goals
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        np.random.shuffle(self.goals)

    def set_end_point(self):
        self.start_point = np.copy(self.end_point)
        pre_x, pre_y, pre_z = self.start_point
        if self.update_direction == 0:
            if self.direction_flag == 0:
                end_x = pre_x + np.random.uniform(low=1.0, high=2.0)
                if end_x > 4.0:
                    self.direction_flag = 1
                    end_x = pre_x - np.random.uniform(low=1.0, high=2.0)
            else:
                end_x = pre_x - np.random.uniform(low=1.0, high=2.0)
                if end_x < -4.0:
                    self.direction_flag = 0
                    end_x = pre_x + np.random.uniform(low=1.0, high=2.0)

            end_y = np.random.uniform(low=-3.5, high=3.5)
            end_z = np.random.uniform(low=2.0, high=3.0)
        else:
            if self.direction_flag == 0:
                end_y = pre_y + np.random.uniform(low=1.0, high=2.0)
                if end_y > 4.0:
                    self.direction_flag = 1
                    end_y = pre_y - np.random.uniform(low=1.0, high=2.0)
            else:
                end_y = pre_y - np.random.uniform(low=1.0, high=2.0)
                if end_y < -4.0:
                    self.direction_flag = 0
                    end_y = pre_y + np.random.uniform(low=1.0, high=2.0)

            end_x = np.random.uniform(low=-3.5, high=3.5)
            end_z = np.random.uniform(low=2.0, high=3.0)

        self.end_point = np.array([end_x, end_y, end_z])
        self.duration_time += np.random.uniform(low=8.0, high=10.0)

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate
            return infos, rewards

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        self.set_end_point()
        self.update_goals()
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]
            env.pos_decay_rate = self.pos_decay_rate

        return infos, rewards

    def reset(self):
        self.cur_start_tick = 0
        self.update_direction = np.random.randint(2)
        self.direction_flag = np.random.randint(2)

        if self.update_direction == 0:
            if self.direction_flag == 0:
                self.init_flag = 3
            else:
                self.init_flag = 1
        elif self.update_direction == 1:
            if self.direction_flag == 0:
                self.init_flag = 2
            else:
                self.init_flag = 0

        self.spawn_flag = self.init_flag
        self.start_point = self.generate_pos(shift_small=1.25, shift_big=2.0, shift_collide=2.5)
        self.end_point = copy.deepcopy(self.start_point)
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.standard_reset(formation_center=self.start_point)


class Scenario_o_inside_obstacles(Scenario_o_dynamic_diff_goal):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.duration_time = 0.0
        self.obstacle_pos = np.array([0.0, 0.0, 2.0])
        self.init_flag = 0
        self.spawn_flag = 0
        self.obst_level = -1
        self.obst_num_in_room = 0
        self.cur_start_tick = 0

    def update_formation_size(self, new_formation_size):
        pass

    def set_end_point(self):
        self.start_point = np.copy(self.end_point)
        obst_x, obst_y = self.obstacle_pos[:2]
        end_x = obst_x + np.random.uniform(low=-0.5, high=0.5)
        end_y = obst_y + np.random.uniform(low=-0.5, high=0.5)

        end_z = self.start_point[2] + np.random.uniform(low=-1.0, high=1.0)
        end_z = np.clip(end_z, a_min=1.0, a_max=4.0)
        self.end_point = np.array([end_x, end_y, end_z])
        self.duration_time += np.random.uniform(low=8.0, high=10.0)

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate

            return infos, rewards

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        if self.obst_level > -1:
            obst_num = len(infos[0]['obstacles'])
            self.obst_num_in_room = min(self.obst_num_in_room, obst_num)
            obst_id = np.random.randint(low=0, high=self.obst_num_in_room)
            self.obstacle_pos = infos[0]['obstacles'][obst_id].pos
        else:
            x, y = np.random.uniform(low=-3.0, high=3.0, size=2)
            z = np.random.uniform(low=1.0, high=3.0)
            self.obstacle_pos = np.array([x, y, z])

        self.set_end_point()
        # Reset formation and related parameters
        self.update_formation_and_relate_param()
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        np.random.shuffle(self.goals)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]
            env.pos_decay_rate = self.pos_decay_rate

        return infos, rewards

    def reset(self, obst_level=-1, obst_level_num_window=4):
        self.cur_start_tick = 0
        self.obst_level = obst_level
        self.obst_num_in_room = np.random.randint(low=self.obst_level - obst_level_num_window + 2,
                                                  high=self.obst_level + 2)
        self.obst_num_in_room = max(1, self.obst_num_in_room)

        self.init_flag = np.random.randint(4)
        self.spawn_flag = self.init_flag
        x, y, z = self.generate_pos(shift_small=1.25, shift_big=2.0, shift_collide=2.5)
        self.start_point = np.array([x, y, z])
        self.end_point = np.array([x, y, z])
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.standard_reset(formation_center=self.start_point)


class Scenario_o_swap_goals(Scenario_o_inside_obstacles):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.duration_time = 0.0
        self.obstacle_pos = np.array([0.0, 0.0, 2.0])
        self.init_flag = 0
        self.spawn_flag = 0
        self.get_obst_flag = False
        self.obst_level = -1
        self.obst_num_in_room = 0
        self.cur_start_tick = 0

    def set_end_point(self):
        self.start_point = np.copy(self.end_point)
        obst_x, obst_y = self.obstacle_pos[:2]
        end_x = obst_x + np.random.uniform(low=-0.5, high=0.5)
        end_y = obst_y + np.random.uniform(low=-0.5, high=0.5)

        end_z = self.start_point[2] + np.random.uniform(low=-1.0, high=1.0)
        end_z = np.clip(end_z, a_min=1.0, a_max=4.0)
        self.end_point = np.array([end_x, end_y, end_z])

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick
        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate

            return infos, rewards

        if self.obst_level > -1:
            if not self.get_obst_flag:
                obst_num = len(infos[0]['obstacles'])
                self.obst_num_in_room = min(self.obst_num_in_room, obst_num)
                obst_id = np.random.randint(low=0, high=self.obst_num_in_room)
                self.obstacle_pos = infos[0]['obstacles'][obst_id].pos
                self.set_end_point()
                self.get_obst_flag = True
        else:
            x, y = np.random.uniform(low=-3.0, high=3.0, size=2)
            z = np.random.uniform(low=1.0, high=3.0)
            self.obstacle_pos = np.array([x, y, z])
            self.set_end_point()

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        self.duration_time += np.random.uniform(low=8.0, high=10.0)
        # Reset formation and related parameters
        self.update_formation_and_relate_param()
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        np.random.shuffle(self.goals)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]
            env.pos_decay_rate = self.pos_decay_rate

        return infos, rewards

    def reset(self, obst_level=-1, obst_level_num_window=4):
        self.cur_start_tick = 0
        self.obst_level = obst_level
        self.obst_num_in_room = np.random.randint(low=self.obst_level - obst_level_num_window + 2,
                                                  high=self.obst_level + 2)
        self.obst_num_in_room = max(1, self.obst_num_in_room)
        self.init_flag = np.random.randint(4)
        self.spawn_flag = self.init_flag
        x, y, z = self.generate_pos(shift_small=1.25, shift_big=2.0, shift_collide=2.5)
        self.start_point = np.array([x, y, z])
        self.end_point = np.array([x, y, z])
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.get_obst_flag = False

        # Reset formation and related parameters
        self.standard_reset(formation_center=self.start_point)


class Scenario_o_swarm_groups(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.group_num = 4
        self.goals_center_list = []
        self.goals_list = []
        # i.e., self.group_num = 3, self.agent_groups = [3, 2, 3]
        self.agent_groups = []
        self.room_dims = room_dims
        self.duration_time = 0.0
        self.quads_mode = quads_mode
        self.env_shuffle_list = np.arange(len(envs))
        self.agent_num = len(envs)
        self.half_room_length = self.room_dims[0] / 2
        self.half_room_width = self.room_dims[1] / 2
        self.cur_start_tick = 0

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.create_formations()
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def get_decent_center(self, x_low, x_high, y_low, y_high):
        x = np.random.uniform(low=x_low, high=x_high)
        y = np.random.uniform(low=y_low, high=y_high)
        for center in self.goals_center_list:
            center_xy = center[:2]
            count = 0
            while np.linalg.norm(np.array([x, y]) - center_xy) <= 1.0:
                if count > 3:
                    break
                x = np.random.uniform(low=x_low, high=x_high)
                y = np.random.uniform(low=y_low, high=y_high)
                count += 1

        return x, y

    def set_agent_groups(self):
        self.agent_groups = [self.agent_num // self.group_num for _ in range(self.group_num)]
        if self.agent_num / self.group_num != self.agent_num // self.group_num:
            surplus_num = self.agent_num % self.group_num
            group_id_list = np.arange(self.group_num)
            np.random.shuffle(group_id_list)
            for i in range(surplus_num):
                self.agent_groups[group_id_list[i]] += 1

    def generate_centers(self, shift_collide=1.5):
        x_low, x_high = -1.0 * self.half_room_length + shift_collide, self.half_room_length - shift_collide
        y_low, y_high = -1.0 * self.half_room_width + shift_collide, self.half_room_width - shift_collide

        for i in range(self.group_num):
            x, y = self.get_decent_center(x_low, x_high, y_low, y_high)
            z = np.random.uniform(low=2.0, high=3.0)
            self.goals_center_list.append(np.array([x, y, z]))

    def create_formations(self):
        tmp_goals = []
        for i in range(self.group_num):
            goals = self.generate_goals(num_agents=self.agent_groups[i], formation_center=self.goals_center_list[i],
                                        layer_dist=self.layer_dist)
            np.random.shuffle(goals)
            tmp_goals.extend(goals)

        self.goals = copy.deepcopy(tmp_goals)
        for i in range(len(self.envs)):
            self.goals[self.env_shuffle_list[i]] = tmp_goals[i]

    def update_goals(self):
        np.random.shuffle(self.goals_center_list)

        self.update_formation_and_relate_param()
        self.create_formations()
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]
            env.pos_decay_rate = self.pos_decay_rate

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate

            return infos, rewards

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        self.update_goals()
        self.duration_time += np.random.uniform(low=10.0, high=15.0)
        return infos, rewards

    def reset(self):
        self.cur_start_tick = 0
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        np.random.shuffle(self.env_shuffle_list)
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Initialize group number
        self.group_num = np.random.randint(3, self.agent_num + 1)  # [low, high)
        self.set_agent_groups()

        # Initialize self.goals_center_list
        self.generate_centers()
        # Reset the formation size and the goals of swarms
        self.create_formations()
        self.formation_center = np.mean(self.goals_center_list, axis=0)


class Scenario_o_ep_rand_bezier(Scenario_o_dynamic_same_goal):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_time = 0.0
        self.init_flag = 0
        self.spawn_flag = 0  # used for init spawn, check quadrotor_single.py
        self.quads_mode = quads_mode
        self.cur_start_tick = 0

    def step(self, infos, rewards, pos):
        # randomly sample new goal pos in free space and have the goal move there following a bezier curve
        tick = self.envs[0].tick

        control_freq = self.envs[0].control_freq
        num_secs = 5
        control_steps = int(num_secs * control_freq)
        t = tick % control_steps
        room_dims = np.array(self.room_dims) - self.formation_size
        # min and max distance the goal can spawn away from its current location. 30 = empirical upper bound on
        # velocity that the drones can handle.
        max_dist = min(30, max(room_dims))
        min_dist = max_dist / 2
        if tick % control_steps == 0 or tick == 1:
            # sample a new goal pos that's within the room boundaries and satisfies the distance constraint
            new_goal_found = False
            while not new_goal_found:
                low, high = np.array([-room_dims[0] / 2, -room_dims[1] / 2, 0]), np.array(
                    [room_dims[0] / 2, room_dims[1] / 2, room_dims[2]])
                # need an intermediate point for a deg=2 curve
                new_pos = np.random.uniform(low=-high, high=high, size=(2, 3)).reshape(3, 2)
                # add some velocity randomization = random magnitude * unit direction
                new_pos = new_pos * np.random.randint(min_dist, max_dist + 1) / np.linalg.norm(new_pos, axis=0)
                new_pos = self.goals[0].reshape(3, 1) + new_pos
                lower_bound = np.expand_dims(low, axis=1)
                upper_bound = np.expand_dims(high, axis=1)
                new_goal_found = (new_pos > lower_bound + 0.5).all() and (
                        new_pos < upper_bound - 0.5).all()  # check bounds that are slightly smaller than the room dims
            nodes = np.concatenate((self.goals[0].reshape(3, 1), new_pos), axis=1)
            nodes = np.asfortranarray(nodes)
            pts = np.linspace(0, 1, control_steps)
            curve = bezier.Curve(nodes, degree=2)
            self.interp = curve.evaluate_multi(pts)
            # self.interp = np.clip(self.interp, a_min=np.array([0,0,0.2]).reshape(3,1), a_max=high.reshape(3,1)) # want goal clipping to be slightly above the floor
        if tick % control_steps != 0 and tick > 1:
            self.goals = np.array([self.interp[:, t] for _ in range(self.num_agents)])

            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]
                env.pos_decay_rate = 1.0

        return infos, rewards

    def reset(self):
        self.cur_start_tick = 0
        self.init_flag = np.random.randint(4)
        self.spawn_flag = self.init_flag
        self.start_point = self.generate_pos(shift_small=1.25, shift_big=2.0, shift_collide=2.5)
        self.end_point = copy.deepcopy(self.start_point)
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.standard_reset(formation_center=self.start_point)


class Scenario_o_uniform_goal_spawn(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_time = 0.0
        self.init_flag = -1
        self.spawn_flag = -1  # used for init spawn, check quadrotor_single.py
        self.quads_mode = quads_mode
        self.obst_num_in_room = 0
        self.pos_decay_rate = 0.999

    def update_formation_size(self, new_formation_size):
        pass

    def generate_pos(self):
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        x = np.random.uniform(low=-1.0 * half_room_length + 0.5, high=half_room_length - 0.5)
        y = np.random.uniform(low=-1.0 * half_room_width + 0.5, high=half_room_width - 0.5)
        z = np.random.uniform(low=1.0, high=2.5)
        return np.array([x, y, z])

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate

            return infos, rewards

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        self.duration_time += self.envs[0].ep_time + 1
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]
            env.pos_decay_rate = self.pos_decay_rate

        return infos, rewards

    def reset(self):
        self.cur_start_tick = 0
        self.start_point = self.generate_pos()
        self.end_point = self.generate_pos()
        self.duration_time = np.random.uniform(low=4.0, high=6.0)
        self.standard_reset(formation_center=self.start_point)


class Scenario_o_uniform_diff_goal_spawn(Scenario_o_uniform_goal_spawn):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_time = 0.0
        self.init_flag = -1
        self.spawn_flag = -1  # used for init spawn, check quadrotor_single.py
        self.quads_mode = quads_mode
        self.obst_num_in_room = 0
        self.pos_decay_rate = 0.999

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center, layer_dist=self.layer_dist)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def update_goals(self):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset goals
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)
        np.random.shuffle(self.goals)

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate

            return infos, rewards

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)

        self.duration_time += self.envs[0].ep_time + 1
        self.update_goals()
        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]
            env.pos_decay_rate = self.pos_decay_rate

        return infos, rewards


class Scenario_o_uniform_swarm_vs_swarm(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.goals_1, self.goals_2 = None, None
        self.goal_center_1, self.goal_center_2 = None, None
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.spawn_flag = -1  # used for init spawn, check quadrotor_single.py
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

        x_1 = np.random.uniform(low=-1.0 * half_room_length + 0.5, high=half_room_length - 0.5)
        y_1 = np.random.uniform(low=-1.0 * half_room_width + 0.5, high=half_room_width - 0.5)

        x_2 = np.random.uniform(low=-1.0 * half_room_length + 0.5, high=half_room_length - 0.5)
        y_2 = np.random.uniform(low=-1.0 * half_room_width + 0.5, high=half_room_width - 0.5)

        z_1, z_2 = np.random.uniform(low=1.0, high=2.5, size=2)

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
            env.pos_decay_rate = self.pos_decay_rate

    def step(self, infos, rewards, pos):
        tick = self.envs[0].tick

        if tick <= int(self.duration_time * self.envs[0].control_freq):
            pos_diff_decay_rate = get_pos_diff_decay_rate(decay_rate=self.pos_decay_rate, tick=tick - self.cur_start_tick)
            for i, env in enumerate(self.envs):
                env.pos_decay_rate = pos_diff_decay_rate

            return infos, rewards

        self.cur_start_tick = int(self.duration_time * self.envs[0].control_freq)
        self.update_goals()
        self.duration_time += self.envs[0].ep_time + 1
        return infos, rewards

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


class Scenario_mix(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation, quads_formation_size, one_pass_per_episode)
        self.room_dims_callback = room_dims_callback

        self.obst_mode = self.envs[0].obstacle_mode
        self.one_pass_per_episode = one_pass_per_episode

        # Once change the parameter here, should also update QUADS_PARAMS_DICT to make sure it is same as run a single scenario
        # key: quads_mode
        # value: 0. formation, 1: [formation_low_size, formation_high_size], 2: episode_time
        if self.obst_mode == 'no_obstacles':
            self.quads_mode_list = QUADS_MODE_LIST
        else:
            self.quads_mode_list = QUADS_MODE_LIST_OBSTACLES
            self.spawn_flag = -1
            self.start_point = np.array([-3.0, -3.0, 2.0])
            self.end_point = np.array([3.0, 3.0, 2.0])
            self.scenario_mode = 'o_dynamic_same_goal'
            self.goals_center_list = []
            self.multi_goals = []
            self.obst_num_in_room = 0

        # actual scenario being used
        self.scenario = None
        self.cur_start_tick = 0

    def name(self):
        """
        :return: the name of the actual scenario used in this episode
        """
        return self.scenario.__class__.__name__

    def step(self, infos, rewards, pos):
        infos, rewards = self.scenario.step(infos=infos, rewards=rewards, pos=pos)
        # This is set for obstacle mode
        self.goals = self.scenario.goals
        self.formation_size = self.scenario.formation_size
        return infos, rewards

    def reset(self, obst_level=-1, obst_level_num_window=2, obst_num=8, max_obst_num=4, obst_level_mode=1, curriculum_min_obst=0):
        self.cur_start_tick = 0
        if obst_level <= -1:
            self.obst_num_in_room = curriculum_min_obst
        else:
            self.obst_num_in_room = np.random.randint(low=obst_level + curriculum_min_obst,
                                                      high=obst_level + obst_level_num_window + curriculum_min_obst)
            self.obst_num_in_room = np.clip(self.obst_num_in_room, a_min=1, a_max=obst_num)

        mode_index = np.random.randint(low=0, high=len(self.quads_mode_list))
        mode = self.quads_mode_list[mode_index]

        # Init the scenario
        self.scenario = create_scenario(quads_mode=mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims, room_dims_callback=self.room_dims_callback,
                                        rew_coeff=self.rew_coeff, quads_formation=self.formation,
                                        quads_formation_size=self.formation_size,
                                        one_pass_per_episode=self.one_pass_per_episode)

        if mode in QUADS_MODE_OBST_INFO_LIST:
            self.scenario.reset(obst_level=obst_level)
        else:
            self.scenario.reset()

        self.goals = self.scenario.goals
        self.formation_size = self.scenario.formation_size
        if self.obst_mode != 'no_obstacles':
            if mode not in QUADS_MODE_GOAL_CENTERS:
                self.spawn_flag = self.scenario.spawn_flag
                self.start_point = self.scenario.start_point
                self.end_point = self.scenario.end_point
            else:
                self.goals_center_list = self.scenario.goals_center_list
                self.multi_goals = self.scenario.goals

            if mode in QUADS_MODE_OBST_INFO_LIST:
                self.obst_num_in_room = self.scenario.obst_num_in_room

            self.scenario_mode = self.scenario.quads_mode
