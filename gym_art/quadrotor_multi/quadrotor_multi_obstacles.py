import numpy as np
from scipy import spatial

from gym_art.quadrotor_multi.octomap_creation import OctTree


class MultiObstacles:
    def __init__(self, num_obstacles=0, room_dims = [10, 10, 10],
                 quad_size=0.046, size=0.0, resolution=0.05, inf_height=True):
        self.num_obstacles = num_obstacles
        self.obstacles = []
        
        self.octree = OctTree(obstacle_size=1.0, room_dims=room_dims, resolution=resolution, inf_height=inf_height)
        self.octree.generate_obstacles(num_obstacles=num_obstacles)
        self.octree.pos_arr
        self.octree.mark_octree()
        self.octree.generateSDF()

    #Test shapes of obs, test collisions
    def reset(self, obs=None, quads_pos=None, start_point=np.array([0., 0., 2.]), end_point=np.array([0., 0., 2.])):
        self.octree.reset()
        self.octree.generate_obstacles(num_obstacles=self.num_obstacles, start_point=start_point, end_point=end_point)

        obstobs = []

        for quad in quads_pos:
            obstobs.append(self.octree.getSurround(quad))
        
        obs = np.concatenate((obs, obstobs), axis=0)

        return obs

    '''def reset(self, set_obstacle=None, formation_size=0.0, goal_central=np.array([0., 0., 2.]), shape='sphere', quads_pos=None, quads_vel=None):
        if set_obstacle is None:
            raise ValueError('set_obstacle is None')

        self.formation_size = formation_size
        self.goal_central = goal_central

        # Reset shape and size
        self.shape = shape
        self.size = np.random.uniform(low=0.15, high=0.5)

        if set_obstacle:
            if self.mode == 'static':
                self.static_obstacle()
            elif self.mode == 'dynamic':
                if self.traj == "mix":
                    traj_id = np.random.randint(low=0, high=len(TRAJ_LIST))
                    self.tmp_traj = TRAJ_LIST[traj_id]
                else:
                    self.tmp_traj = self.traj

                if self.tmp_traj == "electron":
                    self.dynamic_obstacle_electron()
                elif self.tmp_traj == "gravity":
                    # Try 1 + 100 times, make sure initial vel, both vx and vy < 3.0
                    self.dynamic_obstacle_grav()
                    for _ in range(100):
                        if abs(self.vel[0]) > 3.0 or abs(self.vel[1]) > 3.0:
                            self.dynamic_obstacle_grav()
                        else:
                            break
                else:
                    raise NotImplementedError(f'{self.traj} not supported!')
            else:
                raise NotImplementedError(f'{self.mode} not supported!')
        else:
            self.pos = np.array([5., 5., -5.])
            self.vel = np.array([0., 0., 0.])

        obs = self.update_obs(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacle)
        return obs'''

    #Returns observations
    def step(self, obs=None, quads_pos=None, quads_vel=None, set_obstacles=None):
        if set_obstacles is None:
            raise ValueError('set_obstacles is None')

        for quad in quads_pos:
            obstobs = self.octree.getSurround(quad)
            #obst_obs = obstacle.step(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacles[i])
            obs = np.concatenate((obs, obstobs), axis=0)

        return obs

    '''def step(self, quads_pos=None, quads_vel=None, set_obstacle=None):
        if set_obstacle is None:
            raise ValueError('set_obstacle is None')

        if not set_obstacle:
            obs = self.update_obs(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacle)
            return obs

        if self.tmp_traj == 'electron':
            obs = self.step_electron(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacle)
            return obs
        elif self.tmp_traj == 'gravity':
            obs = self.step_gravity(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacle)
            return obs
        else:
            raise NotImplementedError()
            
    
    def update_obs(self, quads_pos, quads_vel, set_obstacle):
        # Add rel_pos, rel_vel, size, shape to obs, shape: num_agents * 10
        if (not set_obstacle) and self.obs_mode == 'absolute':
            rel_pos = self.pos - np.zeros((len(quads_pos), 3))
            rel_vel = self.vel - np.zeros((len(quads_pos), 3))
            obst_size = np.zeros((len(quads_pos), 3))
            obst_shape = np.zeros((len(quads_pos), 1))
        elif (not set_obstacle) and self.obs_mode == 'half_relative':
            rel_pos = self.pos - np.zeros((len(quads_pos), 3))
            rel_vel = self.vel - np.zeros((len(quads_pos), 3))
            obst_size = (self.size / 2) * np.ones((len(quads_pos), 3))
            obst_shape = self.shape_list.index(self.shape) * np.ones((len(quads_pos), 1))
        else:  # False, relative; True
            rel_pos = self.pos - quads_pos
            rel_vel = self.vel - quads_vel
            # obst_size: in xyz axis: radius for sphere, half edge length for cube
            obst_size = (self.size / 2) * np.ones((len(quads_pos), 3))
            obst_shape = self.shape_list.index(self.shape) * np.ones((len(quads_pos), 1))

        obs = np.concatenate((rel_pos, rel_vel, obst_size, obst_shape), axis=1)

        return obs'''

    def collision_detection(self, arm_length, pos_quads=None):

        collision_matrix = np.zeros((len(pos_quads), 1))
        
        for i, quad in enumerate(pos_quads):
            curr = self.octree.SDFDist(quad)
            if curr < arm_length:
                collision_matrix[i] = 1
        
        return collision_matrix

        # Shape: (num_agents, num_obstacles)
        '''collision_matrix = np.zeros((len(pos_quads), self.num_obstacles))

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

        return collision_matrix, drone_collisions, all_collisions, distance_matrix'''

