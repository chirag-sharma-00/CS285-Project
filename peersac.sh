python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_epsilon_1_adv_size_32_adv_dim_4_1000_iter \
--critic_epsilon 1 --advice_net_n_layers 1 --advice_net_size 32 --advice_dim 4 \
--seed 42 --actor_update_frequency 10 \
--num_agents 2 & # --save_params

python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_3_agents_epsilon_1_adv_size_32_adv_dim_4_1000_iter \
--critic_epsilon 1 --advice_net_n_layers 1 --advice_net_size 32 --advice_dim 4 \
--seed 42 --actor_update_frequency 10 \
--num_agents 3 & # --save_params

python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_5_agents_epsilon_1_adv_size_32_adv_dim_4_1000_iter \
--critic_epsilon 1 --advice_net_n_layers 1 --advice_net_size 32 --advice_dim 4 \
--seed 42 --actor_update_frequency 10 \
--num_agents 5 # --save_params