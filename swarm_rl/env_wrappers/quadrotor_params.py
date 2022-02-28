from sample_factory.utils.utils import str2bool


def quadrotors_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_quads',
        hidden_size=256,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
    )


# noinspection PyUnusedLocal
def add_quadrotors_env_args(env, parser):
    p = parser

    p.add_argument('--quads_discretize_actions', default=-1, type=int, help='Discretize actions into N bins for each individual action. Default (-1) means no discretization')
    p.add_argument('--quads_clip_input', default=False, type=str2bool, help='Whether to clip input to ensure it stays relatively small')
    p.add_argument('--quads_effort_reward', default=None, type=float, help='Override default value for effort reward')
    p.add_argument('--quads_episode_duration', default=15.0, type=float, help='Override default value for episode duration')
    p.add_argument('--quads_num_agents', default=8, type=int, help='Override default value for the number of quadrotors')
    p.add_argument('--quads_neighbor_hidden_size', default=256, type=int, help='The hidden size for the neighbor encoder')
    p.add_argument('--quads_neighbor_encoder_type', default='attention', type=str, choices=['attention', 'mean_embed', 'mlp', 'no_encoder'], help='The type of the neighborhood encoder')

    # TODO: better default values for collision rewards
    p.add_argument('--quads_collision_reward', default=0.0, type=float, help='Override default value for quadcol_bin reward, which means collisions between quadrotors')
    p.add_argument('--quads_collision_obstacle_reward', default=0.0, type=float, help='Override default value for quadcol_bin_obst reward, which means collisions between quadrotor and obstacle')
    p.add_argument('--quads_settle', default=False, type=str2bool, help='Use velocity penalty and equal distance rewards when drones are within a certain radius of the goal')
    p.add_argument('--quads_vel_reward_out_range', default=0.8, type=float, help='We only use this parameter when quads_settle=True, the meaning of this parameter is that we would punish the quadrotor if it flies out of the range that we defined')
    p.add_argument('--quads_settle_range_meters', default=1.0, type=float, help='Radius of the sphere around the goal with velocity penalty to help quadrotors stop and settle at the goal')

    p.add_argument('--quads_collision_hitbox_radius', default=2.0, type=float, help='When the distance between two drones are less than N arm_length, we would view them as collide.')
    p.add_argument('--quads_collision_falloff_radius', default=0.0, type=float, help='The falloff radius for the smooth penalty. 0: radius is 0 arm_length, which means we would not add extra penalty except drones collide')
    p.add_argument('--quads_collision_smooth_max_penalty', default=10.0, type=float, help='The upper bound of the collision function given distance among drones')

    p.add_argument('--neighbor_obs_type', default='none', type=str, choices=['none', 'pos_vel', 'pos_vel_size', 'pos_vel_goals', 'pos_vel_goals_ndist_gdist'], help='Choose what kind of obs to send to encoder.')
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')
    p.add_argument('--quads_obstacle_mode', default='no_obstacles', type=str, choices=['no_obstacles', 'static', 'dynamic', 'static_door', 'static_door_fixsize', 'static_random_place_fixsize', 'static_pillar_fixsize'], help='Choose which obstacle mode to run')
    p.add_argument('--quads_obstacle_num', default=0, type=int, help='Choose the number of obstacle(s)')
    p.add_argument('--quads_local_obst_obs', default=-1, type=int, help='Number of obstacles to consider. -1=all obstacles. 0=blind agents, 0<n<num_obstacles-1 = nonzero number of obstacles')
    p.add_argument('--quads_obstacle_type', default='sphere', type=str, choices=['sphere', 'cube', 'cylinder', 'random'], help='Choose the type of obstacle(s)')
    p.add_argument('--quads_obstacle_size', default=0.0, type=float, help='Choose the size of obstacle(s)')
    p.add_argument('--quads_obstacle_traj', default='gravity', type=str, choices=['gravity', 'electron', 'mix'],  help='Choose the type of force to use')
    p.add_argument('--quads_local_obs', default=-1, type=int, help='Number of neighbors to consider. -1=all neighbors. 0=blind agents, 0<n<num_agents-1 = nonzero number of agents')
    p.add_argument('--quads_local_coeff', default=0.0, type=float, help='This parameter is used for the metric of select which drones are the N closest drones.')
    p.add_argument('--quads_local_metric', default='dist_inverse', type=str, choices=['dist', 'dist_inverse'], help='The main part of evaluate the closest drones')

    p.add_argument('--quads_view_mode', default='local', type=str, choices=['local', 'global'], help='Choose which kind of view/camera to use')
    p.add_argument('--quads_adaptive_env', default=False, type=str2bool, help='Iteratively shrink the environment into a tunnel to increase obstacle density based on statistics')

    p.add_argument('--quads_mode', default='static_same_goal', type=str, choices=['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal', 'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'swap_goals', 'dynamic_formations', 'through_hole', 'through_random_obstacles', 'o_dynamic_same_goal', 'o_dynamic_diff_goal', 'o_swarm_vs_swarm', 'o_swap_goals', 'o_dynamic_formations', 'o_ep_lissajous3D', 'o_dynamic_roller', 'o_inside_obstacles', 'o_swarm_groups', 'o_ep_rand_bezier', 'mix', 'o_test', 'o_test_stack', 'o_uniform_goal_spawn', 'o_uniform_diff_goal_spawn', 'o_uniform_swarm_vs_swarm', 'tunnel'], help='Choose which scenario to run. Ep = evader pursuit')
    p.add_argument('--quads_formation', default='circle_horizontal', type=str, choices=['circle_xz_vertical', 'circle_yz_vertical', 'circle_horizontal', 'sphere', 'grid_xz_vertical', 'grid_yz_vertical', 'grid_horizontal'], help='Choose the swarm formation at the goal')
    p.add_argument('--quads_formation_size', default=-1.0, type=float, help='The size of the formation, interpreted differently depending on the formation type. Default (-1) means it is determined by the mode')
    p.add_argument('--room_length', default=10, type=float, help='Length, width, and height dimensions respectively of the quadrotor env')
    p.add_argument('--room_width', default=10, type=float, help='Length, width, and height dimensions respectively of the quadrotor env')
    p.add_argument('--room_height', default=10, type=float, help='Length, width, and height dimensions respectively of the quadrotor env')
    p.add_argument('--quads_obs_repr', default='xyz_vxyz_R_omega', choices=['xyz_vxyz_R_omega', 'xyz_vxyz_R_omega_wall', 'xyz_vxyz_R_omega_floor', 'xyz_vxyz_R_omega_floor_cwallid_cwall'], type=str, help='obs space for drone itself')
    p.add_argument('--replay_buffer_sample_prob', default=0.0, type=float, help='Probability at which we sample from it rather than resetting the env. Set to 0.0 (default) to disable the replay. Set to value in (0.0, 1.0] to use replay buffer')

    p.add_argument('--anneal_start_collision_steps', default=0.0, type=float, help='Anneal collision penalties over this many steps. Default (0.0) is no annealing')
    p.add_argument('--anneal_collision_steps', default=0.0, type=float, help='Anneal collision penalties over this many steps. Default (0.0) is no annealing')
    p.add_argument('--quads_obstacle_obs_mode', default='relative', type=str, choices=['relative', 'absolute', 'half_relative'],  help='Choose the type of force to use')
    p.add_argument('--quads_obstacle_hidden_size', default=32, type=int, help='Choose the type of force to use')
    p.add_argument('--quads_collision_obst_smooth_max_penalty', default=10.0, type=float, help='The upper bound of the collision function given distance among drones')
    p.add_argument('--quads_obst_penalty_fall_off', default=10.0, type=float, help='The upper bound of the collision function given distance among drones')
    p.add_argument('--quads_obst_enable_sim', default=True, type=str2bool, help='That parameter is for testing, True: Enable simulation for collision with obstacles, False: Ignore obstacles even collide with that')
    p.add_argument('--obst_obs_type', default='none', type=str, choices=['none', 'cpoint', 'pos_size', 'posxy_size', 'pos_vel_size', 'pos_vel_size_shape'], help='Choose what kind of obs to send to encoder.')
    p.add_argument('--quads_obst_model_type', default='whole', type=str, help='Whole: self, neighbor, obstacle, we use seperate model deal with them, then concate them together. nei_obst, we concate nei_obst first, then concate with self_encoder')

    p.add_argument('--quads_reward_ep_len', default=True, type=str2bool, help='For each drone, reward scale is same as ep_len or not. For example, ep_len=1600, then rew_crash should equals to 16')
    p.add_argument('--quads_freeze_obst_level', default=False, choices=[True, False], type=str2bool, help='True: Never change obstacles level')
    p.add_argument('--quads_obst_level', default=-1, type=int, help='Obstacle start level, -1 means underearth')
    p.add_argument('--quads_obst_level_mode', default=1, type=int, help='0: Directly move obsatcles from underground to above ground, 1: Gradually change the location of obstacles')
    p.add_argument('--quads_obst_level_crash_min', default=3.0, type=float, help='when crash value >= -1.0 * quads_obst_level_crash_min for 10 continuous episodes, we change level')
    p.add_argument('--quads_obst_level_crash_max', default=4.0, type=float, help='when crash value < -1.0 * quads_obst_level_crash_max for 10 continuous episodes, we change level')

    p.add_argument('--quads_obst_level_col_obst_quad_min', default=2.5, type=float, help='when collision b/w obst & drones < quads_obst_level_col_obst_quad_min for 10 continuous episodes, we change level')
    p.add_argument('--quads_obst_level_col_obst_quad_max', default=2.8, type=float, help='when collision b/w obst & drones > quads_obst_level_col_obst_quad_max for 10 continuous episodes, we change level')

    p.add_argument('--quads_obst_level_col_quad_min', default=0.5, type=float, help='when crash value >= quads_obst_level_col_quad_min for 10 continuous episodes, we change level')
    p.add_argument('--quads_obst_level_col_quad_max', default=0.8, type=float, help='when crash value < quads_obst_level_col_quad_max for 10 continuous episodes, we change level')

    p.add_argument('--quads_obst_level_pos_min', default=90.0, type=float, help='when crash value >= -1.0 * quads_obst_level_pos_min for 10 continuous episodes, we change level')
    p.add_argument('--quads_obst_level_pos_max', default=100.0, type=float, help='when crash value < -1.0 * quads_obst_level_pos_max for 10 continuous episodes, we change level')

    p.add_argument('--quads_obstacle_stack_num', default=4, type=int, help='Choose the number of obstacle(s) per stack')
    p.add_argument('--quads_enable_sim_room', default='none', type=str, help='room: simulate crash with ceiling, wall, floor; ceiling: crash with ceiling; wall: crash with wall; floor: crash with floor')

    p.add_argument('--quads_neighbor_proximity_mode', default=1, type=int, help='0: without dt, 1: with dt, check quad_utils.py, calculate_drone_proximity_penalties')
    p.add_argument('--quads_obst_proximity_mode', default=1, type=int, help='0: without dt, 1: with dt, check quad_utils.py, calculate_obst_drone_proximity_penalties')
    p.add_argument('--quads_obst_inf_height', default=False, type=str2bool, help='True: height == room height, False: customized height')
    p.add_argument('--quads_obst_collision_enable_grace_period', default=False, type=str2bool, help='If use grace period, we only calculate the collision penalty and collision proximity penalty after grace period')

    p.add_argument('--quads_crash_mode', default=0, type=int, help='Check quad_crash_utils.py, crash_params')
    p.add_argument('--quads_clip_floor_vel_mode', default=0, type=int, help='Check quad_crash_utils.py, clip_floor_vel_params')

    p.add_argument('--quads_midreset', default=False, type=str2bool, help='If a drone crashes >= quads_crash_reset_threshold, reset this drone')
    p.add_argument('--quads_crash_reset_threshold', default=200, type=int, help='Threshold of midreset, default: 200 ticks')

    p.add_argument('--quads_obst_midreset', default=False, type=str2bool, help='If a drone collide of obstacles >= quads_obst_col_reset_threshold, reset this drone')
    p.add_argument('--quads_obst_col_reset_threshold', default=1, type=int, help='Threshold of midreset, default: 1')

    p.add_argument('--quads_neighbor_rel_pos_mode', default=0, type=int, choices=[0, 1], help='0: use relative pos between the center point of drones, 1: use relative pos between the closest point of drones')
    p.add_argument('--quads_obst_rel_pos_mode', default=0, type=int, choices=[0, 1, 2], help='0: use relative pos between the center point of drones and obstacles, 1: use relative pos between the closest point of drones and obstacles; 2: clip relative to a specific value if the rel_pos out of a predefined value')
    p.add_argument('--quads_obst_rel_pos_clip_value', default=2.0, type=float, help='when quads_obst_rel_pos_mode=2, clip rel_pos')

    p.add_argument('--quads_print_info', default=False, type=str2bool, help='Print some information for testing')
    p.add_argument('--quads_apply_downwash', default=False, type=str2bool, help='True: apply downwash; False: no downwash')
    p.add_argument('--quads_init_random_state', default=True, type=str2bool, help='True: spawn drones on the air; False: spawn drones on the floor with 0 vel, omega')

    p.add_argument('--quads_normalize_obs', default=False, type=str2bool, help='True: Normalize all observations to [-1, 1]')
    p.add_argument('--quads_one_pass_per_episode', default=False, type=str2bool, help='True: one pass, False: multiple pass')

    p.add_argument('--quads_extra_crash_reward', default=False, type=str2bool, help='True: add extra crash reward')
    p.add_argument('--quads_obst_generation_mode', default='random', type=str, choices=['random', 'cube', 'gaussian'], help='random: randomly place obstacles; cube: sample the center points of obstacles in a gradually increased cube')

    p.add_argument('--quads_use_pos_diff', default=False, type=str2bool, help='Use pos diff as pos metric')

    p.add_argument('--quads_obst_smooth_penalty_mode', default='linear', type=str, choices=['linear', 'square'], help='linear: linear function, square: square function')

    p.add_argument('--quads_larger_obst_encoder', default=False, type=str2bool, help='Use larger obst encoder')
    p.add_argument('--nearest_nbrs', default=0, type=int, help='Set to nonzero when combining obstacles and nbr drones into one observation')
    p.add_argument('--quads_init_from_model', default=False, type=str2bool, help='Init nbr/obst encoder from existing model')
    p.add_argument('--quads_curriculum_min_obst', default=0, type=int, help='Minimum obstacle number if obst_level=-1')
