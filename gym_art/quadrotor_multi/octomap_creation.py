import numpy as np
import math

import octomap

class OctTree:
    def __init__(self, obstacle_size=1.0, room_dims=[10, 10, 10], resolution=0.05, inf_height=True):
        self.resolution = resolution
        self.inf_height = inf_height 
        self.octree = octomap.OcTree(self.resolution)
        self.room_dims = room_dims
        self.half_room_length = self.room_dims[0] / 2
        self.half_room_width = self.room_dims[1] / 2
        self.grid_size = obstacle_size
        self.size = obstacle_size
        self.start_range = np.zeros((2, 2))
        self.end_range = np.zeros((2, 2))
        self.init_box = np.array([[-2.0, -2.0, -0.5 * 2.0], [2.0, 2.0, 1.5 * 2.0]])
        self.cell_centers = [
            (i + (self.grid_size / 2) - self.half_room_length, j + (self.grid_size / 2) - self.half_room_width) for i in
            np.arange(0, self.room_dims[0], self.grid_size) for j in np.arange(0, self.room_dims[1], self.grid_size)]
    
    def reset(self):
        del self.octree
        self.octree = octomap.OcTree(self.resolution)
        return
    
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

    def gaussian_pos(self, obst_id=0, goal_start_point=np.array([-3.0, -2.0, 2.0]), goal_end_point=np.array([3.0, 2.0, 2.0]),
                     y_gaussian_scale=None):
        middle_point = (goal_start_point + goal_end_point)/2

        goal_vector = goal_end_point - goal_start_point
        goal_distance = np.linalg.norm(goal_vector)

        alpha = math.atan2(goal_vector[1], goal_vector[0])

        pos_x = np.random.normal(loc=middle_point[0], scale=goal_distance / 4.0)
        if y_gaussian_scale is None:
            y_gaussian_scale = np.random.uniform(low=0.2, high=0.5)

        pos_y = np.random.normal(loc=middle_point[1], scale=y_gaussian_scale)

        rot_pos_x = middle_point[0] + math.cos(alpha) * (pos_x - middle_point[0]) - math.sin(alpha) * (pos_y - middle_point[1])
        rot_pos_y = middle_point[1] + math.sin(alpha) * (pos_x - middle_point[0]) + math.cos(alpha) * (pos_y - middle_point[1])

        dist = lambda p1, p2: (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        #rot_pos_x, rot_pos_y = min(self.cell_centers, key=lambda coord: dist(coord, (rot_pos_x, rot_pos_y)))

        rot_pos_x = np.clip(rot_pos_x, a_min=-self.half_room_length + self.grid_size, a_max=self.half_room_length - self.grid_size)
        rot_pos_y = np.clip(rot_pos_y, a_min=-self.half_room_width + self.grid_size, a_max=self.half_room_width - self.grid_size)

        pos_xy = np.array([rot_pos_x, rot_pos_y, ])

        collide_start = self.check_pos(pos_xy, self.start_range)
        collide_end = self.check_pos(pos_xy, self.end_range)
        collide_flag = collide_start or collide_end

        return pos_xy, collide_flag
    
    def y_gaussian_generation(self, regen_id=0):
        if regen_id < 3:
            return None

        y_low = 0.13 * regen_id - 0.1
        y_high = y_low * np.random.uniform(low=1.5, high=2.5)
        y_gaussian_scale = np.random.uniform(low=y_low, high=y_high)
        return y_gaussian_scale
    
    def get_pos_no_overlap(self, pos_item, pos_arr, obst_id):
        # In this function, we assume the shape of all obstacles is cube
        # But even if we have this assumption, we can still roughly use it for shapes like cylinder
        if pos_arr.shape[1] == 0:
            return pos_item, False

        overlap_flag = False
        for j in range(len(pos_arr)):
            if np.linalg.norm(pos_item - pos_arr[j][:2]) < 1.5*self.size:
                overlap_flag = True
                break
        return pos_item, overlap_flag

    def generate_obstacles(self, num_obstacles=0, start_point=np.array([-3.0, -2.0, 2.0]), end_point=np.array([3.0, 2.0, 2.0])):
        self.reset()
        self.pos_arr = np.array([[]])
        #TODO Check Scenario Mode
        self.start_range = np.array([start_point[:2] + self.init_box[0][:2],  start_point[:2] + self.init_box[1][:2]])
        self.end_range = np.array([end_point[:2] + self.init_box[0][:2],  end_point[:2] + self.init_box[1][:2]])
        pos_z = 0.5 * self.room_dims[2]
        for i in range(num_obstacles):
            for regen_id in range(20):
                y_gaussian_scale = self.y_gaussian_generation(regen_id=regen_id)
                pos_xy, collide_flag = self.gaussian_pos(obst_id=i, y_gaussian_scale=y_gaussian_scale, goal_start_point=start_point, goal_end_point=end_point)
                pos_item = np.array([pos_xy[0], pos_xy[1]])
                final_pos_item, overlap_flag = pos_item, False
                #self.get_pos_no_overlap(pos_item=pos_item, pos_arr=self.pos_arr, obst_id=i)
                if collide_flag is False and overlap_flag is False:
                    if self.pos_arr.shape[1] == 0:
                        self.pos_arr = np.array([np.append(np.asarray(final_pos_item), pos_z)])
                    self.pos_arr = np.append(self.pos_arr, np.array([np.append(np.asarray(final_pos_item), pos_z)]), axis=0)
                    break

        self.mark_octree()
        self.generateSDF()

        return self.pos_arr
    
    def mark_octree(self):
        range_shape = 0.5 * self.size
        self.octree.updateNode([-1*self.room_dims[0]//2, -1*self.room_dims[1]//2, -1*self.room_dims[2]//2], False)
        self.octree.updateNode([-1 * self.room_dims[0] // 2, -1 * self.room_dims[1] // 2, self.room_dims[2] // 2],
                               False)
        self.octree.updateNode([-1 * self.room_dims[0] // 2, self.room_dims[1] // 2, -1 * self.room_dims[2] // 2],
                               False)
        self.octree.updateNode([-1 * self.room_dims[0] // 2, self.room_dims[1] // 2, self.room_dims[2] // 2],
                               False)
        self.octree.updateNode([self.room_dims[0] // 2, -1 * self.room_dims[1] // 2, -1 * self.room_dims[2] // 2],
                               False)
        self.octree.updateNode([self.room_dims[0] // 2, -1 * self.room_dims[1] // 2, self.room_dims[2] // 2],
                               False)
        self.octree.updateNode([self.room_dims[0] // 2, self.room_dims[1] // 2, -1 * self.room_dims[2] // 2],
                               False)
        self.octree.updateNode([self.room_dims[0] // 2, self.room_dims[1] // 2, self.room_dims[2] // 2],
                               False)

        for item in self.pos_arr:
            for x in np.arange(item[0]-range_shape, item[0]+range_shape+self.resolution, self.resolution):
                for y in np.arange(item[1]-range_shape, item[1]+range_shape+self.resolution, self.resolution):
                    if self.inf_height:
                        for z in np.arange(-1*self.room_dims[2]//2, self.room_dims[2]//2+0.001, self.resolution):
                            if x < self.room_dims[0] and y < self.room_dims[1] and z < self.room_dims[2]:
                                if np.linalg.norm(np.asarray([x, y])-item[:2]) <= self.size/2:
                                    self.octree.updateNode([x, y, z], True)
                    else:
                        for z in np.arange(item[2]-range_shape, item[2]+range_shape+self.resolution, self.resolution):
                            if x < self.room_dims[0] and y < self.room_dims[1] and z < self.room_dims[2]:
                                if np.linalg.norm(np.asarray([x, y])-item[:2]) <= self.size/2:
                                    self.octree.updateNode([x, y, z], True)

    def generateSDF(self):
        self.octree.dynamicEDT_generate(2,
                                        np.array([-1*self.room_dims[0]/2, -1*self.room_dims[1]/2, -1*self.room_dims[2]/2]), 
                                        np.array([self.room_dims[0]/2, self.room_dims[1]/2, self.room_dims[2]/2]))
        self.octree.dynamicEDT_update(True)

    def SDFDist(self, p):
        return self.octree.dynamicEDT_getDistance(p)

    def getSurround(self, p):
        state = []
        for i in np.arange(p[0]-self.resolution, p[0]+self.resolution+0.0001, self.resolution):
            for j in np.arange(p[1]-self.resolution, p[1]+self.resolution+0.0001, self.resolution):
                for k in np.arange(p[2]-self.resolution, p[2]+self.resolution+0.0001, self.resolution):
                    state.append(self.SDFDist(np.array([i, j, k])))

        state = np.array(state)
        return state