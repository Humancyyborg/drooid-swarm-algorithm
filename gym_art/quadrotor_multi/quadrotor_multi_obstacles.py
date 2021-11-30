import numpy as np
from scipy import spatial

from gym_art.quadrotor_multi.quadrotor_single_obstacle import SingleObstacle
from gym_art.quadrotor_multi.quad_obstacle_utils import OBSTACLES_SHAPE_LIST, STATIC_OBSTACLE_DOOR

EPS = 1e-6


class MultiObstacles:
    def __init__(self, mode='no_obstacles', num_obstacles=0, max_init_vel=1., init_box=2.0, dt=0.005,
                 quad_size=0.046, shape='sphere', size=0.0, traj='gravity', obs_mode='relative', num_local_obst=-1,
                 obs_type='pos_size', drone_env=None, level=-1):
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
            pos_arr = self.generate_pos_by_level(level=level)

        for i in range(self.num_obstacles):
            obstacle = SingleObstacle(max_init_vel=max_init_vel, init_box=init_box, mode=mode, shape=shape, size=size,
                                      quad_size=quad_size, dt=dt, traj=traj, obs_mode=obs_mode, index=i,
                                      obs_type=obs_type, all_pos_arr=pos_arr)
            self.obstacles.append(obstacle)

    def reset(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=None, formation_size=0.0, goal_central=np.array([0., 0., 2.]),
              level=-1):
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
            pos_arr = self.generate_pos_by_level(level=level)

        for i, obstacle in enumerate(self.obstacles):
            obst_obs = obstacle.reset(set_obstacle=set_obstacles[i], formation_size=formation_size,
                                      goal_central=goal_central, shape=shape_list[i], quads_pos=quads_pos,
                                      quads_vel=quads_vel, new_pos=pos_arr[i])
            all_obst_obs.append(obst_obs)

        all_obst_obs = np.stack(all_obst_obs)
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
        obs = self.concat_obstacle_obs(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel, all_obst_obs=all_obst_obs)
        return obs

    def collision_detection(self, pos_quads=None, set_obstacles=None):
        if set_obstacles is None:
            raise ValueError('set_obstacles is None')

        # Shape: (num_agents, num_obstacles)
        collision_matrix = np.zeros((len(pos_quads), self.num_obstacles))

        for i, obstacle in enumerate(self.obstacles):
            if set_obstacles[i]:
                col_arr = obstacle.collision_detection(pos_quads=pos_quads)
                collision_matrix[:, i] = col_arr

        # check which drone collide with obstacle(s)
        drone_collisions = []
        all_collisions = []
        col_w1 = np.where(collision_matrix >= 1)
        for i, val in enumerate(col_w1[0]):
            drone_collisions.append(col_w1[0][i])
            all_collisions.append((col_w1[0][i], col_w1[1][i]))

        obst_positions = np.stack([self.obstacles[i].pos for i in range(self.num_obstacles)])
        distance_matrix = spatial.distance_matrix(x=pos_quads, y=obst_positions)

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
                rel_pos, rel_vel = self.get_rel_pos_vel_item(quad_pos=quads_pos[i], quad_vel=quads_vel[i], indices=indices[i])
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

        obst_stack_num = int(self.num_obstacles / 4)
        if obst_stack_num == 1:
            if level <= 8:
                pos_x = np.random.uniform(low=-1.0, high=1.0)
                pos_y = np.random.uniform(low=-1.0, high=1.0)
            else:
                pos_x = np.random.uniform(low=-3.0, high=3.0)
                pos_y = np.random.uniform(low=-3.0, high=3.0)

            level_z = np.clip(level, -1, 8)
            pos_z_bottom = 0.5 * self.size * level_z - self.size * self.num_obstacles

            # Add pos
            for i in range(self.num_obstacles):
                tmp_pos_arr = np.array([pos_x, pos_y, pos_z_bottom + self.size * (0.5 + i)])
                pos_arr.append(tmp_pos_arr)

        elif obst_stack_num == 2:
            pos_x_0 = np.random.uniform(low=-2.0, high=-0.5)
            pos_y_0 = np.random.uniform(low=-1.0, high=1.0)

            pos_x_1 = np.random.uniform(low=0.5, high=2.0)
            pos_y_1 = np.random.uniform(low=-1.0, high=1.0)

            level_z = np.clip(level, -1, 8)
            pos_z_bottom = 0.5 * self.size * level_z - self.size * (self.num_obstacles/2)

            # Add pos
            for i in range(int(self.num_obstacles/2)):
                tmp_pos_arr_0 = np.array([pos_x_0, pos_y_0, pos_z_bottom + self.size * (0.5 + i)])
                tmp_pos_arr_1 = np.array([pos_x_1, pos_y_1, pos_z_bottom + self.size * (0.5 + i)])
                pos_arr.append(tmp_pos_arr_0)
                pos_arr.append(tmp_pos_arr_1)

        return pos_arr
