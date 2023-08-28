import unittest
from unittest import TestCase

from sample_factory.envs.create_env import create_env
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, is_module_available

from swarm_rl.train import register_swarm_components, parse_swarm_cfg


def numba_available():
    return is_module_available('numba')


def run_multi_quadrotor_env(env_name, cfg):
    env = create_env(env_name, cfg=cfg)
    env.reset()
    for i in range(100):
        obs, r, term, trunc, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

    n_frames = 1000
    env = create_env(env_name, cfg=cfg)
    env.reset()

    timing = Timing()
    with timing.timeit('step'):
        for i in range(n_frames):
            obs, r, term, trunc, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

    log.debug('Time %s, FPS %.1f', timing, n_frames * env.num_agents / timing.step)
    env.close()


class TestQuads(TestCase):
    def test_quad_multi_env(self):
        register_swarm_components()

        env_name = 'quadrotor_multi'
        experiment_name = 'test_multi'
        cfg = parse_swarm_cfg(argv=["--algo=APPO", f"--env={env_name}", f"--experiment={experiment_name}"])
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        run_multi_quadrotor_env(env_name, cfg)

    @unittest.skipUnless(numba_available(), 'Numba is not installed')
    def test_quad_multi_env_with_numba(self):
        register_swarm_components()

        env_name = 'quadrotor_multi'
        experiment_name = 'test_numba'
        cfg = parse_swarm_cfg(argv=["--algo=APPO", f"--env={env_name}", f"--experiment={experiment_name}"])
        cfg.quads_use_numba = True
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        run_multi_quadrotor_env(env_name, cfg)
