import numpy as np

from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix


def test_calculate_collision_matrix():
    positions = np.ones((8, 3))
    positions[7][0] = 3
    positions[7][1] = 3
    positions[7][2] = 6
    collision_threshold = 0.2
    num_agents = 8

    item_num = int(num_agents * (num_agents - 1) / 2)
    test_drone_col_matrix, test_curr_drone_collisions, test_distance_matrix = \
        calculate_collision_matrix(positions=positions, collision_threshold=collision_threshold)

    true_drone_col_matrix = -1000 * np.ones(len(positions))
    true_curr_drone_collisions = -1000 * np.ones((item_num, 2))
    true_distance_matrix = -1000 * np.ones((item_num, 3))
    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            true_distance_matrix[count] = [i, j, np.linalg.norm(positions[i] - positions[j])]
            if np.linalg.norm(positions[i] - positions[j]) <= collision_threshold:
                true_drone_col_matrix[i] = 1
                true_drone_col_matrix[j] = 1
                true_curr_drone_collisions[count] = [i, j]
            count += 1

    test_curr_drone_collisions = test_curr_drone_collisions.astype(int)
    test_curr_drone_collisions = np.delete(test_curr_drone_collisions, np.unique(
        np.where(test_curr_drone_collisions == [-1000, -1000])[0]), axis=0)

    true_curr_drone_collisions = true_curr_drone_collisions.astype(int)
    true_curr_drone_collisions = np.delete(true_curr_drone_collisions, np.unique(
        np.where(true_curr_drone_collisions == [-1000, -1000])[0]), axis=0)

    assert test_drone_col_matrix.all() == true_drone_col_matrix.all()

    for i in range(len(test_curr_drone_collisions)):
        if test_curr_drone_collisions[i] not in true_curr_drone_collisions:
            raise ValueError

    assert test_distance_matrix.all() == true_distance_matrix.all()

    # print('drone_col_matrix:    ', drone_col_matrix)
    # print('curr_drone_collisions:    ', curr_drone_collisions)
    # print('distance_matrix:    ', distance_matrix)

    return


def unit_test():
    test_calculate_collision_matrix()
    print('Pass unit test!')
    return


if __name__ == "__main__":
    unit_test()
