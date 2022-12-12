python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 400 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_v2 \
--critic_epsilon 0.3 --advice_dim 4
--seed 42 --actor_update_frequency 10 \
--num_agents 2 --critic_version 2
# --save_params