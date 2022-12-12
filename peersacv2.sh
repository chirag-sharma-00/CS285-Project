# Default 2 agents
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_2agents \
--critic_epsilon 0.3 --advice_dim 4
--seed 42 --actor_update_frequency 10 \
--num_agents 2 --critic_version 2 # --save_params

# Default 3 agents
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_3agents \
--critic_epsilon 0.3 --advice_dim 4
--seed 43 --actor_update_frequency 10 \
--num_agents 3 --critic_version 2 # --save_params

# Default 5 agents
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_5agents \
--critic_epsilon 0.3 --advice_dim 4
--seed 44 --actor_update_frequency 10 \
--num_agents 5 --critic_version 2 # --save_params

# 3 agents, advice_dim=8
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_3agents_advdim8 \
--critic_epsilon 0.3 --advice_dim 8
--seed 45 --actor_update_frequency 10 \
--num_agents 3 --critic_version 2 # --save_params

# 3 agents, critic_epsilon=0.7
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_3agents_eps0.7 \
--critic_epsilon 0.7 --advice_dim 4
--seed 46 --actor_update_frequency 10 \
--num_agents 3 --critic_version 2 # --save_params

# 3 agents, init_temperature=0.5
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.5 --exp_name peersac_v2_3agents_temp0.5 \
--critic_epsilon 0.3 --advice_dim 4
--seed 47 --actor_update_frequency 10 \
--num_agents 3 --critic_version 2 # --save_params

