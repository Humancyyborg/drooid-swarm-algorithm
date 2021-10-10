import numpy as np

OBSTACLES_SHAPE_LIST = ['sphere', 'cube']
Y_VALUE = 0.0
STATIC_OBSTACLE_DOOR = np.array([
                                [-1.0, Y_VALUE, 2.5], [0.0, Y_VALUE, 2.5], [1.0, Y_VALUE, 2.5],
                                [-0.65, Y_VALUE, 1.5],                      [0.65, Y_VALUE, 1.5],
                                [-1.0, Y_VALUE, 0.5], [0.0, Y_VALUE, 0.5], [1.0, Y_VALUE, 0.5]])