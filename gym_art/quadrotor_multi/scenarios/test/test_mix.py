import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario
from gym_art.quadrotor_multi.scenarios.utils import QUADS_MODE_LIST_SINGLE, QUADS_MODE_LIST, \
    QUADS_MODE_LIST_OBSTACLES

from gym_art.quadrotor_multi.scenarios.static_same_goal import Scenario_static_same_goal
from gym_art.quadrotor_multi.scenarios.dynamic_diff_goal import Scenario_dynamic_diff_goal
from gym_art.quadrotor_multi.scenarios.dynamic_formations import Scenario_dynamic_formations
from gym_art.quadrotor_multi.scenarios.dynamic_same_goal import Scenario_dynamic_same_goal
from gym_art.quadrotor_multi.scenarios.ep_lissajous3D import Scenario_ep_lissajous3D
from gym_art.quadrotor_multi.scenarios.ep_rand_bezier import Scenario_ep_rand_bezier
from gym_art.quadrotor_multi.scenarios.run_away import Scenario_run_away
from gym_art.quadrotor_multi.scenarios.static_diff_goal import Scenario_static_diff_goal
from gym_art.quadrotor_multi.scenarios.static_same_goal import Scenario_static_same_goal
from gym_art.quadrotor_multi.scenarios.swap_goals import Scenario_swap_goals
from gym_art.quadrotor_multi.scenarios.swarm_vs_swarm import Scenario_swarm_vs_swarm
from gym_art.quadrotor_multi.scenarios.obstacles.o_uniform_diff_goal_spawn import Scenario_o_uniform_diff_goal_spawn
from gym_art.quadrotor_multi.scenarios.obstacles.o_uniform_same_goal_spawn import Scenario_o_uniform_same_goal_spawn
from gym_art.quadrotor_multi.scenarios.obstacles.o_uniform_swarm_vs_swarm import Scenario_o_uniform_swarm_vs_swarm


def create_scenario(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                    quads_formation_size):
    cls = eval('Scenario_' + quads_mode)
    scenario = cls(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                   quads_formation_size)
    return scenario


class Scenario_mix_test(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                 quads_formation_size):
        super().__init__(quads_mode, envs, num_agents, room_dims, room_dims_callback, rew_coeff, quads_formation,
                         quads_formation_size)

        self.room_dims_callback = room_dims_callback

        # Once change the parameter here, should also update QUADS_PARAMS_DICT to make sure it is same as run a
        # single scenario key: quads_mode value: 0. formation, 1: [formation_low_size, formation_high_size],
        # 2: episode_time
        if num_agents == 1:
            self.quads_mode_list = QUADS_MODE_LIST_SINGLE
        elif num_agents > 1 and not envs[0].use_obstacles:
            quads_mode_list = np.array(QUADS_MODE_LIST)
            self.quads_mode_list = quads_mode_list[[0, 3, 6, 7, 8]]
        elif envs[0].use_obstacles:
            self.quads_mode_list = QUADS_MODE_LIST_OBSTACLES

            # Add parameters
            self.start_point = np.array([-3.0, -3.0, 2.0])
            self.end_point = np.array([3.0, 3.0, 2.0])
            self.scenario_mode = 'o_dynamic_same_goal'

        # actual scenario being used
        self.scenario = None
        self.counter = 0

    def name(self):
        """
        :return: the name of the actual scenario used in this episode
        """
        return self.scenario.__class__.__name__

    def step(self, infos, pos):
        infos = self.scenario.step(infos=infos, pos=pos)
        # This is set for obstacle mode
        self.goals = self.scenario.goals
        self.formation_size = self.scenario.formation_size
        return infos

    def reset(self):
        mode = self.quads_mode_list[self.counter]
        self.counter = (self.counter + 1) % len(self.quads_mode_list)

        # Init the scenario
        self.scenario = create_scenario(quads_mode=mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims, room_dims_callback=self.room_dims_callback,
                                        rew_coeff=self.rew_coeff, quads_formation=self.formation,
                                        quads_formation_size=self.formation_size)

        self.scenario.reset()
        self.goals = self.scenario.goals
        self.formation_size = self.scenario.formation_size

        if self.envs[0].use_obstacles:
            self.start_point = self.scenario.start_point
            self.end_point = self.scenario.end_point
