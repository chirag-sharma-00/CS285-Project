python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
--n_iter 40000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_epsilon_0.6_adv_size_32_adv_dim_4_40k_iter \
--critic_epsilon 0.6 --advice_net_n_layers 1 --advice_net_size 32 --advice_dim 4 \
--seed 6 --actor_update_frequency 10 \
--num_agents 2 & # --save_params

python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
--n_iter 40000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_epsilon_0.6_adv_size_32_adv_dim_8_40k_iter \
--critic_epsilon 0.6 --advice_net_n_layers 1 --advice_net_size 32 --advice_dim 8 \
--seed 6 --actor_update_frequency 10 \
--num_agents 2 & # --save_params

python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
--n_iter 40000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_epsilon_0.6_adv_size_32_adv_dim_16_40k_iter \
--critic_epsilon 0.6 --advice_net_n_layers 1 --advice_net_size 32 --advice_dim 16 \
--seed 6 --actor_update_frequency 10 \
--num_agents 2 & # --save_params

python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
--n_iter 40000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_epsilon_0.6_adv_size_16_adv_dim_4_40k_iter \
--critic_epsilon 0.6 --advice_net_n_layers 1 --advice_net_size 16 --advice_dim 4 \
--seed 6 --actor_update_frequency 10 \
--num_agents 2 & # --save_params

python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
--n_iter 40000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_epsilon_0.6_adv_size_64_adv_dim_4_40k_iter \
--critic_epsilon 0.6 --advice_net_n_layers 1 --advice_net_size 64 --advice_dim 4 \
--seed 6 --actor_update_frequency 10 \
--num_agents 2 # --save_params

# python cs285/scripts/run_sac_peer_experiment.py \
# --env_name HalfCheetah-v4 --ep_len 150 \
# --discount 0.99 --scalar_log_freq 1500 \
# --n_iter 40000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
# -lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_epsilon_0_adv_size_64_adv_dim_4_40k_iter \
# --critic_epsilon 0 --advice_net_n_layers 1 --advice_net_size 64 --advice_dim 4 \
# --seed 6 --actor_update_frequency 10 \
# --num_agents 2 & # --save_params

# python cs285/scripts/run_sac_peer_experiment.py \
# --env_name HalfCheetah-v4 --ep_len 150 \
# --discount 0.99 --scalar_log_freq 1500 \
# --n_iter 40000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
# -lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_epsilon_0.1_adv_size_64_adv_dim_4_40k_iter \
# --critic_epsilon 0.1 --advice_net_n_layers 1 --advice_net_size 64 --advice_dim 4 \
# --seed 6 --actor_update_frequency 10 \
# --num_agents 2 & # --save_params

# python cs285/scripts/run_sac_peer_experiment.py \
# --env_name HalfCheetah-v4 --ep_len 150 \
# --discount 0.99 --scalar_log_freq 1500 \
# --n_iter 40000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
# -lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_epsilon_0.3_adv_size_64_adv_dim_4_40k_iter \
# --critic_epsilon 0.3 --advice_net_n_layers 1 --advice_net_size 64 --advice_dim 4 \
# --seed 6 --actor_update_frequency 10 \
# --num_agents 2 & # --save_params

# python cs285/scripts/run_sac_peer_experiment.py \
# --env_name HalfCheetah-v4 --ep_len 150 \
# --discount 0.99 --scalar_log_freq 1500 \
# --n_iter 40000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
# -lr 0.0003 --init_temperature 0.1 --exp_name sac_peer_2_agents_epsilon_0.6_adv_size_64_adv_dim_4_40k_iter \
# --critic_epsilon 0.6 --advice_net_n_layers 1 --advice_net_size 64 --advice_dim 4 \
# --seed 6 --actor_update_frequency 10 \
# --num_agents 2 # --save_params