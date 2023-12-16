from sample_factory.utils.utils import str2bool


def quadrotors_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_quads',
        rnn_size=256,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
    )


# noinspection PyUnusedLocal
def add_quadrotors_env_args(env, parser):
    p = parser

    # Quadrotor features
    p.add_argument('--quads_num_agents', default=8, type=int, help='Override default value for the number of quadrotors')
    p.add_argument('--quads_obs_repr', default='xyz_vxyz_R_omega', type=str,
                   choices=['xyz_vxyz_R_omega', 'xyz_vxyz_R_omega_floor', 'xyz_vxyz_R_omega_wall'],
                   help='obs space for quadrotor self')
    p.add_argument('--quads_episode_duration', default=15.0, type=float,
                   help='Override default value for episode duration')
    p.add_argument('--quads_encoder_type', default="attention", type=str, help='The type of the neighborhood encoder')
    p.add_argument('--quads_obs_acc_his', default=False, type=str2bool, help='If use history acc in obs or not')
    p.add_argument('--quads_obs_acc_his_num', default=3, type=int, help='The number of history acceleration')

    # Neighbor
    # Neighbor Features
    p.add_argument('--quads_neighbor_visible_num', default=-1, type=int, help='Number of neighbors to consider. -1=all '
                                                                          '0=blind agents, '
                                                                          '0<n<num_agents-1 = nonzero number of agents')
    p.add_argument('--quads_neighbor_obs_type', default='none', type=str,
                   choices=['none', 'pos_vel'], help='Choose what kind of obs to send to encoder.')

    # # Neighbor Encoder
    p.add_argument('--quads_neighbor_hidden_size', default=256, type=int,
                   help='The hidden size for the neighbor encoder')
    p.add_argument('--quads_neighbor_encoder_type', default='attention', type=str,
                   choices=['attention', 'mean_embed', 'mlp', 'no_encoder'],
                   help='The type of the neighborhood encoder')

    # # Neighbor Collision Reward
    p.add_argument('--quads_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin reward, which means collisions between quadrotors')
    p.add_argument('--quads_collision_hitbox_radius', default=2.0, type=float,
                   help='When the distance between two drones are less than N arm_length, we would view them as '
                        'collide.')
    p.add_argument('--quads_collision_falloff_radius', default=-1.0, type=float,
                   help='The falloff radius for the smooth penalty. -1.0: no smooth penalty')
    p.add_argument('--quads_collision_smooth_max_penalty', default=10.0, type=float,
                   help='The upper bound of the collision function given distance among drones')

    # Obstacle
    # # Obstacle Features
    p.add_argument('--quads_use_obstacles', default=False, type=str2bool, help='Use obstacles or not')
    p.add_argument('--quads_obstacle_obs_type', default='none', type=str,
                   choices=['none', 'octomap'], help='Choose what kind of obs to send to encoder.')
    p.add_argument('--quads_obst_density', default=0.2, type=float, help='Obstacle density in the map')
    p.add_argument('--quads_obst_size', default=1.0, type=float, help='The radius of obstacles')
    p.add_argument('--quads_obst_spawn_area', nargs='+', default=[8.0, 8.0], type=float,
                   help='The spawning area of obstacles')
    p.add_argument('--quads_domain_random', default=False, type=str2bool, help='Use domain randomization or not')
    p.add_argument('--quads_obst_density_random', default=False, type=str2bool, help='Enable obstacle density randomization or not')
    p.add_argument('--quads_obst_density_min', default=0.05, type=float,
                   help='The minimum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_density_max', default=0.2, type=float,
                   help='The maximum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_size_random', default=False, type=str2bool, help='Enable obstacle size randomization or not')
    p.add_argument('--quads_obst_size_min', default=0.3, type=float,
                   help='The minimum obstacle size when enabling domain randomization')
    p.add_argument('--quads_obst_size_max', default=0.6, type=float,
                   help='The maximum obstacle size when enabling domain randomization')

    # # Obstacle Encoder
    p.add_argument('--quads_obst_hidden_size', default=256, type=int, help='The hidden size for the obstacle encoder')
    p.add_argument('--quads_obst_encoder_type', default='mlp', type=str, help='The type of the obstacle encoder')

    # # Obstacle Collision Reward
    p.add_argument('--quads_obst_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin_obst reward, which means collisions between quadrotor '
                        'and obstacles')

    # Aerodynamics
    # # Downwash
    p.add_argument('--quads_use_downwash', default=False, type=str2bool, help='Apply downwash or not')

    # Numba Speed Up
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')

    # Scenarios
    p.add_argument('--quads_mode', default='static_same_goal', type=str,
                   choices=['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal',
                            'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'swap_goals', 'dynamic_formations',
                            'mix', 'o_uniform_same_goal_spawn', 'o_random',
                            'o_dynamic_diff_goal', 'o_dynamic_same_goal', 'o_diagonal', 'o_static_same_goal',
                            'o_static_diff_goal', 'o_swap_goals'], help='Choose which scenario to run. ep = evader pursuit')

    # Room
    p.add_argument('--quads_room_dims', nargs='+', default=[10., 10., 10.], type=float,
                   help='Length, width, and height dimensions respectively of the quadrotor env')

    # Replay Buffer
    p.add_argument('--replay_buffer_sample_prob', default=0.0, type=float,
                   help='Probability at which we sample from it rather than resetting the env. Set to 0.0 (default) '
                        'to disable the replay. Set to value in (0.0, 1.0] to use replay buffer')

    # Annealing
    p.add_argument('--anneal_collision_steps', default=0.0, type=float,
                   help='Anneal collision penalties over this many steps. Default (0.0) is no annealing')

    # Rendering
    p.add_argument('--quads_view_mode', nargs='+', default=['topdown', 'chase', 'global'],
                   type=str, choices=['topdown', 'chase', 'side', 'global', 'corner0', 'corner1', 'corner2', 'corner3', 'topdownfollow'],
                   help='Choose which kind of view/camera to use')
    p.add_argument('--quads_render', default=False, type=bool, help='Use render or not')
    p.add_argument('--visualize_v_value', action='store_true', help="Visualize v value map")

    # Controller
    # # Reward coefficient
    p.add_argument('--quads_cost_rl_sbc', default=0.1, type=float,
                   help='cost coeff of the difference between rl acc outputs and sbc acc outputs')
    p.add_argument('--quads_cost_rl_mellinger', default=0.1, type=float,
                   help='cost coeff of the difference between rl acc outputs and mellinger acc outputs')
    p.add_argument('--quads_sbc_boundary', default=0.1, type=float, help='sbc boundary in reward function')
    p.add_argument('--quads_cost_pos', default=1.0, type=float, help='cost coeff of the position cost')
    p.add_argument('--quads_cost_crash', default=1.0, type=float, help='cost coeff of the crash cost')
    p.add_argument('--quads_cost_act_change', default=1.0, type=float, help='cost coeff of the action change cost')
    p.add_argument('--quads_cost_cbf_agg', default=1.0, type=float, help='cost coeff of the aggressiveness cost')
    p.add_argument('--quads_cost_enable_extra', default=False, type=str2bool,
                   help='enable cost for action, effort, and orientation')

    # # Reward annealing for: 1) quads_cost_rl_sbc 2) quads_cost_rl_mellinger
    p.add_argument('--quads_anneal_safe_start_steps', default=0.0, type=float, help='Start annealing')
    p.add_argument('--quads_anneal_safe_total_steps', default=0.0, type=float, help='Total annealing steps')

    # Enable sbc
    p.add_argument('--quads_enable_sbc', default=True, type=str2bool, help='Whether to use sbc or not ')

    # # CBF aggressiveness annealing for:
    # # 1) quads_max_neighbor_aggressive 2) quads_max_obst_aggressive
    p.add_argument('--cbf_agg_anneal_steps', default=0.0, type=float,
                   help='Anneal steps for cbf aggressiveness, neighbor and obst. Default (0.0) is no annealing')

    # # Aggressiveness
    p.add_argument('--quads_max_neighbor_aggressive', default=5.0, type=float, help='maximum aggressive')
    p.add_argument('--quads_max_obst_aggressive', default=5.0, type=float, help='maximum aggressive')
    p.add_argument('--quads_max_room_aggressive', default=0.2, type=float, help='maximum aggressive')

    # # Others
    p.add_argument('--quads_sbc_radius', default=0.05, type=float, help='sbc sensing radius')
    p.add_argument('--quads_neighbor_range', default=2.0, type=float, help='Consider other drones in this range')
    p.add_argument('--quads_obst_range', default=2.0, type=float, help='Consider obstacles in this range')
    p.add_argument('--quads_max_acc', default=2.0, type=float, help='maximum acceleration')
