"""
Main script for training a swarm of quadrotors with SampleFactory

"""

import sys

from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env
from swarm_rl.env_wrappers.quadrotor_params import add_quadrotors_env_args, quadrotors_override_defaults
from swarm_rl.models.quad_multi_model import register_models


def register_custom_components():
    global_env_registry().register_env(
        env_name_prefix='quadrotor_',
        make_env_func=make_quadrotor_env,
        add_extra_params_func=add_quadrotors_env_args,
        override_default_params_func=quadrotors_override_defaults,
    )

    register_models()


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_args(evaluation=False)
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
