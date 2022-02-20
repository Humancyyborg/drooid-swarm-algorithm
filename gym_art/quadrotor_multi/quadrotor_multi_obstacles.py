import numpy as np

from gym_art.quadrotor_multi.quad_scenarios_utils import QUADS_MODE_MULTI_GOAL_CENTER, QUADS_MODE_GOAL_CENTERS
from gym_art.quadrotor_multi.quadrotor_single_obstacle import SingleObstacle
from gym_art.quadrotor_multi.quad_obstacle_utils import OBSTACLES_SHAPE_LIST, STATIC_OBSTACLE_DOOR
import math
EPS = 1e-6


class MultiObstacles:
    def __init__(self, mode='no_obstacles', num_obstacles=0, max_init_vel=1., init_box=2.0, dt=0.005,
                 quad_size=0.046, shape='sphere', size=0.0, traj='gravity', obs_mode='relative', num_local_obst=-1,
                 obs_type='pos_size', drone_env=None, level=-1, stack_num=4, level_mode=0, inf_height=False,
                 room_dims=(10.0, 10.0, 10.0), rel_pos_mode=0, rel_pos_clip_value=2.0, obst_level_num_window=4,
                 obst_generation_mode='random', obst_change_step=1.0):
        if 'static_door' in mode:
            self.num_obstacles = len(STATIC_OBSTACLE_DOOR)
        else:
            self.num_obstacles = num_obstacles

        self.obstacles = []
        self.shape = shape
        self.shape_list = OBSTACLES_SHAPE_LIST
        self.num_local_obst = num_local_obst
        self.size = size
        self.mode = mode
        self.drone_env = drone_env
        self.stack_num = stack_num
        self.level_mode = level_mode
        self.room_height = drone_env.room_box[1][2]
        self.inf_height = inf_height
        self.room_dims = room_dims
        self.half_room_length = self.room_dims[0] / 2
        self.half_room_width = self.room_dims[1] / 2
        self.start_range = np.zeros((2, 2))
        self.end_range = np.zeros((2, 2))
        self.start_range_list = []
        self.scenario_mode = None
        self.rel_pos_clip_value = rel_pos_clip_value
        self.obst_level_num_window = obst_level_num_window
        self.obst_num_in_room = 0
        self.obst_generation_mode = obst_generation_mode
        self.change_step = obst_change_step
        self.grid_size = 1.0
        self.max_obst_num = int((self.half_room_length - 0.5 * self.grid_size) / self.change_step)
        # self.counter = 0
        # self.counter_list = []

        pos_arr = []
        if 'static_random_place' in mode:
            num_rest_obst = num_obstacles
            room_box = drone_env.room_box
            pos_block_arr = []
            for i in range(num_obstacles):
                if num_rest_obst <= 0:
                    break
                obst_num_in_block = np.random.randint(low=1, high=min(4, num_rest_obst + 1))  # [low, high)
                num_rest_obst -= obst_num_in_block
                # 4 = 2 * the inital pos box for drones,
                # 0.5 * size is to make sure the init pos of drones not inside of obst
                pos_x = np.random.uniform(low=room_box[0][0] + 0.5 * size, high=room_box[1][0] - 0.5 * size)
                pos_y = np.random.uniform(low=room_box[0][1] + 2 + 0.5 * size, high=room_box[1][1] - 2 - 0.5 * size)
                obst_pos_xy = np.array([pos_x, pos_y])

                # Check collision
                for block_item_pos in pos_block_arr:
                    # As long as dist > sqrt(2) * size, obstacles will not overlap with each other
                    # extra 0.6 size area for drones to fly through the gap area
                    block_xy = block_item_pos[:2]
                    if np.linalg.norm(obst_pos_xy - block_xy) < size:
                        for try_time in range(3):
                            pos_x = np.random.uniform(low=room_box[0][0] + 0.5 * size, high=room_box[1][0] - 0.5 * size)
                            pos_y = np.random.uniform(low=room_box[0][1] + 2 + 0.5 * size, high=room_box[1][1] - 2 - 0.5 * size)
                            obst_pos_xy = np.array([pos_x, pos_y])
                            if np.linalg.norm(obst_pos_xy - block_xy) >= size:
                                break
                # Add pos
                for obst_id in range(obst_num_in_block):
                    tmp_pos_arr = np.array([pos_x, pos_y, size * (0.5 + obst_id)])
                    pos_arr.append(tmp_pos_arr)

                pos_block_arr.append(np.array([pos_x, pos_y, 0.5 * size]))
        elif 'static_pillar' in mode:
            pos_arr = np.zeros((num_obstacles, 3))

        for i in range(self.num_obstacles):
            obstacle = SingleObstacle(max_init_vel=max_init_vel, init_box=init_box, mode=mode, shape=shape, size=size,
                                      quad_size=quad_size, dt=dt, traj=traj, obs_mode=obs_mode, index=i,
                                      obs_type=obs_type, all_pos_arr=pos_arr, inf_height=inf_height, room_dims=room_dims,
                                      rel_pos_mode=rel_pos_mode, rel_pos_clip_value=rel_pos_clip_value)
            self.obstacles.append(obstacle)

    def reset(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=None, formation_size=0.0,
              goal_central=np.array([0., 0., 2.]), level=-1, goal_start_point=np.array([-3.0, -3.0, 2.0]),
              goal_end_point=np.array([3.0, 3.0, 2.0]), scenario_mode='o_dynamic_same_goal', obst_num_in_room=0):

        self.scenario_mode = scenario_mode
        self.obst_num_in_room = obst_num_in_room

        # self.counter = 0
        if self.num_obstacles <= 0:
            return obs
        if set_obstacles is None:
            raise ValueError('set_obstacles is None')

        if self.shape == 'random':
            shape_list = self.get_shape_list()
        else:
            shape_list = [self.shape for _ in range(self.num_obstacles)]
            shape_list = np.array(shape_list)

        all_obst_obs = []
        pos_arr = [None for _ in range(self.num_obstacles)]
        if 'static_pillar' in self.mode:
            if self.inf_height:
                pos_arr = self.generate_inf_pos_by_level(level=level, goal_start_point=goal_start_point,
                                                         goal_end_point=goal_end_point, scenario_mode=scenario_mode)
            else:
                pos_arr = self.generate_pos_by_level(level=level)

        for i, obstacle in enumerate(self.obstacles):
            obst_obs = obstacle.reset(set_obstacle=set_obstacles[i], formation_size=formation_size,
                                      goal_central=goal_central, shape=shape_list[i], quads_pos=quads_pos,
                                      quads_vel=quads_vel, new_pos=pos_arr[i])
            all_obst_obs.append(obst_obs)

        all_obst_obs = np.stack(all_obst_obs)

        if self.num_local_obst != 0:
            obs = self.concat_obstacle_obs(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel, all_obst_obs=all_obst_obs)
        return obs

    def step(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=None):
        if set_obstacles is None:
            raise ValueError('set_obstacles is None')

        all_obst_obs = []
        for i, obstacle in enumerate(self.obstacles):
            obst_obs = obstacle.step(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacles[i])
            all_obst_obs.append(obst_obs)

        all_obst_obs = np.stack(all_obst_obs)
        if self.num_local_obst != 0:
            obs = self.concat_obstacle_obs(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel, all_obst_obs=all_obst_obs)
        return obs

    def collision_detection(self, pos_quads=None, set_obstacles=None):
        if set_obstacles is None:
            raise ValueError('set_obstacles is None')

        # Shape: (num_agents, num_obstacles)
        collision_matrix = np.zeros((len(pos_quads), self.num_obstacles))
        distance_matrix = np.zeros((len(pos_quads), self.num_obstacles))

        for i, obstacle in enumerate(self.obstacles):
            if set_obstacles[i]:
                col_arr, dist_arr = obstacle.collision_detection(pos_quads=pos_quads)
                collision_matrix[:, i] = col_arr
                distance_matrix[:, i] = dist_arr

        # check which drone collide with obstacle(s)
        drone_collisions = []
        all_collisions = []
        col_w1 = np.where(collision_matrix >= 1)
        for i, val in enumerate(col_w1[0]):
            drone_collisions.append(col_w1[0][i])
            all_collisions.append((col_w1[0][i], col_w1[1][i]))

        return collision_matrix, drone_collisions, all_collisions, distance_matrix

    def get_shape_list(self):
        all_shapes = np.array(self.shape_list)
        shape_id_list = np.random.randint(low=0, high=len(all_shapes), size=self.num_obstacles)
        shape_list = all_shapes[shape_id_list]
        return shape_list

    def get_rel_pos_vel_item(self, quad_pos=None, quad_vel=None, indices=None):
        if indices is None:
            # if not specified explicitly, consider all obstacles
            indices = [j for j in range(self.num_obstacles)]

        pos_neighbor = np.stack([self.obstacles[j].pos for j in indices])
        vel_neighbor = np.stack([self.obstacles[j].vel for j in indices])
        # Shape of pos_rel and vel_vel: num_obst * 3
        pos_rel = pos_neighbor - quad_pos
        vel_rel = vel_neighbor - quad_vel
        return pos_rel, vel_rel

    def neighborhood_indices(self, quads_pos, quads_vel):
        """"Return a list of closest obstacles for each drone in the swarm"""
        # indices of all obstacles
        num_quads = len(quads_pos)
        indices = [[i for i in range(self.num_obstacles)] for _ in range(num_quads)]
        indices = np.array(indices)

        if self.num_local_obst == self.num_obstacles or self.num_local_obst == -1:
            return indices
        elif 1 <= self.num_local_obst < self.num_obstacles:
            close_neighbor_indices = []

            for i in range(num_quads):
                # Shape: num_obstacles * 3
                rel_pos, rel_vel = self.get_rel_pos_vel_item(quad_pos=quads_pos[i], quad_vel=quads_vel[i],
                                                             indices=indices[i])
                rel_dist = np.linalg.norm(rel_pos, axis=1)
                rel_dist = np.maximum(rel_dist, 0.01)
                rel_pos_unit = rel_pos / rel_dist[:, None]

                # new relative distance is a new metric that combines relative position and relative velocity
                # F = alpha * distance + (1 - alpha) * dot(normalized_direction_to_other_drone, relative_vel)
                # the smaller the new_rel_dist, the closer the drones
                new_rel_dist = rel_dist + 0.1 * np.sum(rel_pos_unit * rel_vel, axis=1)

                rel_pos_index = new_rel_dist.argsort()
                rel_pos_index = rel_pos_index[:self.num_local_obst]
                close_neighbor_indices.append(indices[i][rel_pos_index])

            # Shape: num_quads * num_local_obst
            return close_neighbor_indices
        else:
            raise RuntimeError("Incorrect number of neigbors")

    def extend_obs_space(self, obs, closest_indices, all_obst_obs):
        obs_neighbors = []
        # len(closest_obsts) = num_agents
        # Shape of closest_obsts: num_agents * num_local_obst
        # Change shape of all_obst_obs (num_obst * num_agents * obs_shape) -> (num_agents * num_obst * obs_shape)
        all_obst_obs = all_obst_obs.swapaxes(0, 1)
        for i in range(len(closest_indices)):
            # all_obst_obs[i][j] means select n closest obstacles given drone i
            # Shape of cur_obsts_obs: (num_local_obst, obst_obs)
            cur_obsts_obs = np.array([all_obst_obs[i][j] for j in closest_indices[i]])
            # Append: (num_local_obst * obst_obs)
            obs_neighbors.append(cur_obsts_obs.reshape(-1))

        obs_neighbors = np.stack(obs_neighbors)
        obs_ext = np.concatenate((obs, obs_neighbors), axis=1)

        return obs_ext

    def concat_obstacle_obs(self, obs, quads_pos, quads_vel, all_obst_obs):
        # Shape all_obst_obs: num_obstacles * num_agents * obst_obs
        # Shape: indices: num_agents * num_local_obst
        indices = self.neighborhood_indices(quads_pos=quads_pos, quads_vel=quads_vel)
        obs_ext = self.extend_obs_space(obs, closest_indices=indices, all_obst_obs=all_obst_obs)
        return obs_ext

    def generate_pos_by_level(self, level=-1):
        pos_arr = []

        obst_stack_num = int(self.num_obstacles / self.stack_num)
        level_split = 2.0 * self.stack_num
        if obst_stack_num == 1:
            if level <= level_split:
                pos_x = np.random.uniform(low=-1.0, high=1.0)
                pos_y = np.random.uniform(low=-1.0, high=1.0)
            else:
                pos_x = np.random.uniform(low=-2.0, high=2.0)
                pos_y = np.random.uniform(low=-2.0, high=2.0)

            pos_z_bottom = 0.0
            if self.level_mode == 0:
                if level >= 0:
                    pos_z_bottom = 0.0
                else:
                    pos_z_bottom = self.size * (-0.5 - self.num_obstacles)
            elif self.level_mode == 1:
                level_z = np.clip(level, -1, level_split)
                pos_z_bottom = 0.5 * self.size * level_z - self.size * self.num_obstacles

            # Add pos
            for i in range(self.num_obstacles):
                tmp_pos_arr = np.array([pos_x, pos_y, pos_z_bottom + self.size * (0.5 + i)])
                pos_arr.append(tmp_pos_arr)

        elif obst_stack_num == 2:
            pos_x_0 = np.random.uniform(low=-2.0, high=-0.5)
            pos_y_0 = np.random.uniform(low=-2.0, high=2.0)

            pos_x_1 = np.random.uniform(low=0.5, high=2.0)
            pos_y_1 = np.random.uniform(low=-2.0, high=2.0)

            pos_z_bottom = 0.0
            if self.level_mode == 0:
                if level >= 0:
                    pos_z_bottom = 0.0
                else:
                    pos_z_bottom = self.size * (-0.5 - self.stack_num)
            elif self.level_mode == 1:
                level_z = np.clip(level, -1, level_split)
                pos_z_bottom = 0.5 * self.size * level_z - self.size * self.stack_num

            # Add pos
            for i in range(int(self.num_obstacles / 2)):
                tmp_pos_arr_0 = np.array([pos_x_0, pos_y_0, pos_z_bottom + self.size * (0.5 + i)])
                tmp_pos_arr_1 = np.array([pos_x_1, pos_y_1, pos_z_bottom + self.size * (0.5 + i)])
                pos_arr.append(tmp_pos_arr_0)
                pos_arr.append(tmp_pos_arr_1)

        return pos_arr

    def check_pos(self, pos_xy, goal_range):
        min_pos = goal_range[0] - np.array([0.5 * self.size, 0.5 * self.size])
        max_pos = goal_range[1] + np.array([0.5 * self.size, 0.5 * self.size])
        closest_point = np.maximum(min_pos, np.minimum(pos_xy, max_pos))
        closest_dist = np.linalg.norm(pos_xy - closest_point)
        if closest_dist <= 0.25:
            # obstacle collide with the spawn range of drones
            return True
        else:
            return False

    def random_pos(self, obst_id=0, goal_start_point=np.array([-3.0, -2.0, 2.0]), goal_end_point=np.array([3.0, 2.0, 2.0])):
        pos_x = round(np.random.uniform(low=-1.0 * self.half_room_length + 1.0, high=self.half_room_length - 1.0))
        pos_y = round(np.random.uniform(low=-1.0 * self.half_room_width + 1.0, high=self.half_room_width - 1.0))
        pos_xy = np.array([pos_x, pos_y]) + self.grid_size / 2

        if self.scenario_mode not in QUADS_MODE_GOAL_CENTERS:
            collide_start = self.check_pos(pos_xy, self.start_range)
            collide_end = self.check_pos(pos_xy, self.end_range)
            collide_flag = collide_start or collide_end
        else:
            collide_flag = False
            for start_range in self.start_range_list:
                collide_start = self.check_pos(pos_xy, start_range)
                if collide_start:
                    collide_flag = True
                    break

        return pos_xy, collide_flag

    def gaussian_pos(self, obst_id=0, goal_start_point=np.array([-3.0, -2.0, 2.0]), goal_end_point=np.array([3.0, 2.0, 2.0])):
        middle_point = (goal_start_point + goal_end_point)/2

        goal_vector = goal_end_point - goal_start_point
        alpha = math.atan2(goal_vector[1], goal_vector[0])

        gaussian_scale = np.random.uniform(low=0.25 * self.half_room_length, high=0.5 * self.half_room_length, size=2)

        pos_x = np.random.normal(loc=middle_point[0], scale=gaussian_scale[0])

        pos_y = np.random.normal(loc=middle_point[1], scale=gaussian_scale[1])

        rot_pos_x = middle_point[0] + math.cos(alpha) * (pos_x - middle_point[0]) - math.sin(alpha) * (pos_y - middle_point[1])
        rot_pos_y = middle_point[1] + math.sin(alpha) * (pos_x - middle_point[0]) + math.cos(alpha) * (pos_y - middle_point[1])

        rot_pos_x = round(rot_pos_x)
        rot_pos_y = round(rot_pos_y)

        rot_pos_x = np.clip(rot_pos_x, a_min=-self.half_room_length + 1.0, a_max=self.half_room_length - 1.0)
        rot_pos_y = np.clip(rot_pos_y, a_min=-self.half_room_width + 1.0, a_max=self.half_room_width - 1.0)

        pos_xy = np.array([rot_pos_x, rot_pos_y]) + self.grid_size / 2

        if self.scenario_mode not in QUADS_MODE_GOAL_CENTERS:
            collide_start = self.check_pos(pos_xy, self.start_range)
            collide_end = self.check_pos(pos_xy, self.end_range)
            collide_flag = collide_start or collide_end
        else:
            collide_flag = False
            for start_range in self.start_range_list:
                collide_start = self.check_pos(pos_xy, start_range)
                if collide_start:
                    collide_flag = True
                    break

        return pos_xy, collide_flag

    def cube_pos(self, obst_id=0, goal_start_point=np.array([-3.0, -2.0, 2.0]), goal_end_point=np.array([3.0, 2.0, 2.0])):
        if obst_id == 0:
            pos_x = np.random.uniform(low=-1.0 * self.change_step, high=self.change_step)
            pos_y = np.random.uniform(low=-1.0 * self.change_step, high=self.change_step)
            pos_xy = np.array([pos_x, pos_y])
            return pos_xy, False

        if obst_id + 1 > self.max_obst_num:
            high_id = max(self.max_obst_num, 2)
            tmp_id = np.random.randint(low=1, high=high_id)  # [low, high)
            area_half = self.change_step * (tmp_id + 1)
        else:
            area_half = self.change_step * (obst_id + 1)

        pre_half = area_half - self.change_step

        area_id = np.random.randint(low=0, high=4)  # [0, 4)
        if area_id == 0:
            x_area = [-area_half, pre_half]
            y_area = [pre_half, area_half]
        elif area_id == 1:
            x_area = [pre_half, area_half]
            y_area = [-pre_half, area_half]
        elif area_id == 2:
            x_area = [-pre_half, area_half]
            y_area = [-area_half, -pre_half]
        elif area_id == 3:
            x_area = [-area_half, -pre_half]
            y_area = [-area_half, pre_half]
        else:
            raise NotImplementedError(f'area_id: {area_id} is not supported!')

        pos_x = round(np.random.uniform(low=x_area[0], high=x_area[1]))
        pos_y = round(np.random.uniform(low=y_area[0], high=y_area[1]))

        pos_xy = np.array([pos_x, pos_y]) + self.grid_size / 2

        if self.scenario_mode not in QUADS_MODE_GOAL_CENTERS:
            collide_start = self.check_pos(pos_xy, self.start_range)
            collide_end = self.check_pos(pos_xy, self.end_range)
            collide_flag = collide_start or collide_end
        else:
            collide_flag = False
            for start_range in self.start_range_list:
                collide_start = self.check_pos(pos_xy, start_range)
                if collide_start:
                    collide_flag = True
                    break

        return pos_xy, collide_flag

    def generate_pos(self, obst_id=0, goal_start_point=np.array([-3.0, -2.0, 2.0]), goal_end_point=np.array([3.0, 2.0, 2.0])):
        if self.obst_generation_mode == 'random':
            pos_generation = self.random_pos
        elif self.obst_generation_mode == 'gaussian':
            pos_generation = self.gaussian_pos
        elif self.obst_generation_mode == 'cube':
            if self.obst_num_in_room <= self.max_obst_num:
                pos_generation = self.cube_pos
            else:
                pos_generation = self.random_pos
        else:
            raise NotImplementedError(f'obst_generation_mode: {self.obst_generation_mode} is not supported!')

        pos_xy, collide_flag = pos_generation(obst_id=obst_id, goal_start_point=goal_start_point,
                                              goal_end_point=goal_end_point)


        return pos_xy, collide_flag

    def get_pos_no_overlap(self, pos_item, pos_arr, obst_id):
        # In this function, we assume the shape of all obstacles is cube
        # But even if we have this assumption, we can still roughly use it for shapes like cylinder
        if len(pos_arr) == 0:
            return pos_item, False

        if self.shape not in ['cube', 'cylinder']:
            raise NotImplementedError(f'{self.shape} not supported!')

        if self.inf_height:
            range_shape = np.array([0.5 * self.size, 0.5 * self.size, 0.5 * self.room_dims[2]])
        else:
            range_shape = 0.5 * self.size

        min_pos = pos_item - range_shape
        max_pos = pos_item + range_shape
        min_pos_arr = pos_arr - range_shape
        max_pos_arr = pos_arr + range_shape

        overlap_flag = False
        for j in range(len(pos_arr)):
            if all(min_pos < max_pos_arr[j]) and all(max_pos > min_pos_arr[j]):
                overlap_flag = True
                break

        return pos_item, overlap_flag

    def generate_inf_pos_by_level(self, level=-1, goal_start_point=np.array([-3.0, -3.0, 2.0]),
                                  goal_end_point=np.array([3.0, 3.0, 2.0]), scenario_mode='o_dynamic_same_goal'):

        init_box_range = self.drone_env.init_box_range
        pos_z = 0.5 * self.room_height

        outbox_pos_item = np.array([self.half_room_length + self.size + self.rel_pos_clip_value,
                                    self.half_room_width + self.size + self.rel_pos_clip_value,
                                    pos_z])

        if level <= -1:
            pos_arr = np.array([outbox_pos_item for _ in range(self.num_obstacles)])
            return pos_arr

        # Based on room_dims [10, 10, 10]
        if scenario_mode not in QUADS_MODE_GOAL_CENTERS:
            self.start_range = np.array([goal_start_point[:2] + init_box_range[0][:2],
                                         goal_start_point[:2] + init_box_range[1][:2]])

            if scenario_mode in QUADS_MODE_MULTI_GOAL_CENTER:
                self.end_range = np.array([goal_end_point[:2] + init_box_range[0][:2],
                                           goal_end_point[:2] + init_box_range[1][:2]])
            else:
                self.end_range = np.array([goal_end_point[:2] + np.array([-0.5, -0.5]),
                                           goal_end_point[:2] + np.array([0.5, 0.5])])
        else:
            for start_point in goal_start_point:
                start_range = np.array([start_point[:2] + init_box_range[0][:2],
                                        start_point[:2] + init_box_range[1][:2]])

                self.start_range_list.append(start_range)

        if self.level_mode == 0:
            if level > -1:
                pos_arr = []
                for i in range(self.num_obstacles):
                    for regen_id in range(20):
                        pos_xy, collide_flag = self.generate_pos(obst_id=i, goal_start_point=goal_start_point,
                                                                 goal_end_point=goal_end_point)
                        pos_item = np.array([pos_xy[0], pos_xy[1], pos_z])
                        final_pos_item, overlap_flag = self.get_pos_no_overlap(pos_item=pos_item, pos_arr=pos_arr, obst_id=i)
                        if collide_flag is False and overlap_flag is False:
                            pos_arr.append(final_pos_item)
                            break

                    if len(pos_arr) <= i:
                        pos_arr.append(final_pos_item)

                return np.array(pos_arr)

        self.obst_num_in_room = min(self.obst_num_in_room, self.num_obstacles)
        pos_arr = []
        for i in range(self.obst_num_in_room):
            for regen_id in range(20):
                pos_xy, collide_flag = self.generate_pos(obst_id=i, goal_start_point=goal_start_point,
                                                         goal_end_point=goal_end_point)
                pos_item = np.array([pos_xy[0], pos_xy[1], pos_z])
                final_pos_item, overlap_flag = self.get_pos_no_overlap(pos_item=pos_item, pos_arr=pos_arr, obst_id=i)
                if collide_flag is False and overlap_flag is False:
                    pos_arr.append(final_pos_item)
                    break

            if len(pos_arr) <= i:
                pos_arr.append(final_pos_item)

        for i in range(self.num_obstacles - self.obst_num_in_room):
            pos_arr.append(outbox_pos_item)

        # self.counter_list.append(self.counter)
        # print('counter: ', self.counter)
        # print('mean: ', np.mean(self.counter_list))
        # print('list counter: ', self.counter_list)
        # print('pos_arr: ', np.array(pos_arr))

        return np.array(pos_arr)
