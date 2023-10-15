import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


def cross(direction, tmp_shift_dir):
    return direction[0] * tmp_shift_dir[1] - direction[1] * tmp_shift_dir[0]


class Scenario_dynamic_same_goal(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.take_off_goal = np.array([2., 2., 2.])
        self.final_goal = np.array([2., 2., 2.])
        self.label = 'left'
        self.subgoals = None
        self.take_off_tick = 100
        self.goal_change_freq_tick = 50
        # TODO: If the pos of the quadrotor is in 0.2 meters of the goal pos for 1s, reset

    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        tick = self.envs[0].tick
        if tick <= self.take_off_tick:
            return
        else:
            tick = tick - self.take_off_tick
            if tick % self.goal_change_freq_tick == 0 and tick > 0:
                cur_subgoal_id = int(tick // self.goal_change_freq_tick)
                cur_subgoal_id = min(cur_subgoal_id, len(self.subgoals)-1)
                cur_subgoal = self.subgoals[cur_subgoal_id]
                for i, env in enumerate(self.envs):
                    env.goal = cur_subgoal
                self.goals = [cur_subgoal for _ in range(self.num_agents)]

        return

    def reset(self, start_point=None):
        # Take off
        z = np.random.uniform(0.5, 2.5)
        self.take_off_goal = np.array(start_point)
        self.take_off_goal[2] = z
        take_off_vel = np.random.uniform(low=0.5, high=1.0)
        self.take_off_tick = int(100 * z / take_off_vel)

        # Final goal
        x, y = np.random.uniform(-3., 3., size=(2,))
        z = np.random.uniform(0.5, 2.5)
        self.final_goal = np.array([x, y, z])

        # Generate sequence
        random_select = np.random.randint(low=0, high=2)
        rel_pos = self.final_goal[:2] - self.take_off_goal[:2]
        dist = np.linalg.norm(rel_pos)
        direction = rel_pos / (dist + 1e-6 if dist == 0.0 else dist)

        if random_select == 0:
            shift_dir = np.array([-1.0 * direction[1], direction[0]])
        else:
            shift_dir = np.array([direction[1], -1.0 * direction[0]])

        if cross(direction, shift_dir) >= 0:
            self.label = 'left'
        else:
            self.label = 'right'

        waypoints_gap = np.random.uniform(low=0.2, high=0.5)

        waypoints_num = int(dist // waypoints_gap)
        waypoints_num = max(waypoints_num, 3)
        # Given label, generate sequences
        self.subgoals = np.linspace(start=self.take_off_goal, stop=self.final_goal, num=waypoints_num)

        for subgoal in self.subgoals[:-1]:
            shift_mag = np.random.uniform(low=0.2, high=2.0)
            subgoal[:2] += shift_dir * shift_mag
            subgoal[2] = self.take_off_goal[2] + (self.final_goal[2] - self.take_off_goal[2]) / waypoints_num

        # subgoal change freq in ticks, 100 ticks = 1s
        subgoal_vel = np.random.uniform(low=1.0, high=2.0)
        self.goal_change_freq_tick = int(100 * waypoints_gap / subgoal_vel)

        for i, env in enumerate(self.envs):
            env.goal = self.take_off_goal
        self.goals = [self.take_off_goal for _ in range(self.num_agents)]
