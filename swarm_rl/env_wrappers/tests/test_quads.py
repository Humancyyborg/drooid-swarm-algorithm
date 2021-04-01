import unittest
from unittest import TestCase

from sample_factory.algorithms.utils.arguments import default_cfg
from sample_factory.envs.create_env import create_env
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, is_module_available

from swarm_rl.train import register_custom_components


def numba_available():
    return is_module_available('numba')


def run_multi_quadrotor_env(env_name, cfg):
    env = create_env(env_name, cfg=cfg)
    env.reset()
    for i in range(100):
        obs, r, d, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

    n_frames = 1000
    env = create_env(env_name, cfg=cfg)
    env.reset()

    timing = Timing()
    with timing.timeit('step'):
        for i in range(n_frames):
            obs, r, d, info = env.step([env.action_space.sample() for _ in range(env.num_agents)])

    log.debug('Time %s, FPS %.1f', timing, n_frames * env.num_agents / timing.step)
    env.close()


class TestQuads(TestCase):
    def test_quad_env(self):
        register_custom_components()

        env_name = 'quadrotor_single'
        cfg = default_cfg(env=env_name)
        self.assertIsNotNone(create_env(env_name, cfg=cfg))

        env = create_env(env_name, cfg=cfg)
        obs = env.reset()

        n_frames = 4000

        timing = Timing()
        with timing.timeit('step'):
            for i in range(n_frames):
                obs, r, d, info = env.step(env.action_space.sample())
                if d:
                    env.reset()

        log.debug('Time %s, FPS %.1f', timing, n_frames / timing.step)

    def test_quad_multi_env(self):
        register_custom_components()

        env_name = 'quadrotor_multi'
        cfg = default_cfg(env=env_name)
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        run_multi_quadrotor_env(env_name, cfg)

    @unittest.skipUnless(numba_available(), 'Numba is not installed')
    def test_quad_multi_env_with_numba(self):
        register_custom_components()

        env_name = 'quadrotor_multi'
        cfg = default_cfg(env=env_name)
        cfg.quads_use_numba = True
        self.assertIsNotNone(create_env(env_name, cfg=cfg))
        run_multi_quadrotor_env(env_name, cfg)
