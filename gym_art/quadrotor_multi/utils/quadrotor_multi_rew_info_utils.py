import numpy as np


def get_collision_reward(num_agents, emergent_collisions, rew_coeff=2.0):
    rew_raw = np.zeros(num_agents)
    if emergent_collisions.any():
        rew_raw[emergent_collisions] = -1.0
    rew = rew_coeff * rew_raw

    return rew_raw, rew


def get_collision_reward_room(num_agents, emergent_collisions_floor, emergent_collisions_walls,
                              emergent_collisions_ceiling, rew_coeff_floor=2.5, rew_coeff_walls=2.0,
                              rew_coeff_ceiling=2.0):

    rew_raw_floor, rew_floor = get_collision_reward(
        num_agents=num_agents, emergent_collisions=emergent_collisions_floor, rew_coeff=rew_coeff_floor)

    rew_raw_walls, rew_walls = get_collision_reward(
        num_agents=num_agents, emergent_collisions=emergent_collisions_walls, rew_coeff=rew_coeff_walls)

    rew_raw_ceiling, rew_ceiling = get_collision_reward(
        num_agents=num_agents, emergent_collisions=emergent_collisions_ceiling, rew_coeff=rew_coeff_ceiling)

    return rew_raw_floor, rew_floor, rew_raw_walls, rew_walls, rew_raw_ceiling, rew_ceiling


def set_collision_rewards(rewards, rew_floor, rew_walls, rew_ceiling,
                          rew_collisions_neighbor, rew_collision_proximity_neighbor,
                          rew_collisions_obst=None, rew_collision_proximity_obst=None):

    if rew_collisions_obst is None:
        rew_collisions_obst = []
    if rew_collision_proximity_obst is None:
        rew_collision_proximity_obst = []

    rewards += rew_floor + rew_walls + rew_ceiling + rew_collisions_neighbor + rew_collision_proximity_neighbor

    if len(rew_collisions_obst) > 0 and len(rew_collision_proximity_obst) > 0:
        rewards += rew_collisions_obst + rew_collision_proximity_obst

    return rewards


def set_collision_infos(infos, rew_floor, rew_walls, rew_ceiling, rew_collisions_neighbor,
                        rew_collision_proximity_neighbor, rew_collisions_obst=None, rew_collision_proximity_obst=None):

    if rew_collisions_obst is None:
        rew_collisions_obst = []
    if rew_collision_proximity_obst is None:
        rew_collision_proximity_obst = []

    for i, info in enumerate(infos):
        info["rewards"]["rew_floor"] = rew_floor[i]
        info["rewards"]["rew_walls"] = rew_walls[i]
        info["rewards"]["rew_ceiling"] = rew_ceiling[i]

        info["rewards"]["rew_col_neighbor"] = rew_collisions_neighbor[i]
        info["rewards"]["rew_collision_prox_neighbor"] = rew_collision_proximity_neighbor[i]

    if len(rew_collisions_obst) > 0 and len(rew_collision_proximity_obst) > 0:
        for i, info in enumerate(infos):
            info["rewards"]["rew_collisions_obst"] = rew_collisions_obst[i]
            info["rewards"]["rew_collision_proximity_obst"] = rew_collision_proximity_obst[i]

    return infos


def set_collision_rewards_infos(rewards, infos, rew_floor, rew_walls, rew_ceiling, rew_collisions_neighbor,
                                rew_collision_proximity_neighbor, rew_collisions_obst=None,
                                rew_collision_proximity_obst=None):

    rewards = set_collision_rewards(rewards=rewards, rew_floor=rew_floor, rew_walls=rew_walls, rew_ceiling=rew_ceiling,
                                    rew_collisions_neighbor=rew_collisions_neighbor,
                                    rew_collision_proximity_neighbor=rew_collision_proximity_neighbor,
                                    rew_collisions_obst=rew_collisions_obst,
                                    rew_collision_proximity_obst=rew_collision_proximity_obst)

    infos = set_collision_infos(infos=infos, rew_floor=rew_floor, rew_walls=rew_walls, rew_ceiling=rew_ceiling,
                                rew_collisions_neighbor=rew_collisions_neighbor,
                                rew_collision_proximity_neighbor=rew_collision_proximity_neighbor,
                                rew_collisions_obst=rew_collisions_obst,
                                rew_collision_proximity_obst=rew_collision_proximity_obst)

    return rewards, infos
