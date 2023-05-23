python -m swarm_rl.enjoy --env=quadrotor_multi \
--replay_buffer_sample_prob=0 \
--anneal_collision_steps=0 \
--train_dir=./train_dir \
--experiment=obstacles_multi_test \
--eval_deterministic=True \
--quads_render=True
