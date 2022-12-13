# eps0.6 2 agents
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_2agents_eps0.6 \
--critic_epsilon 0.6 --advice_dim 4
--seed 42 --actor_update_frequency 10 \
--num_agents 2 --critic_version 2 # --save_params

# eps0.6 3 agents
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_3agents_eps0.6 \
--critic_epsilon 0.6 --advice_dim 4
--seed 43 --actor_update_frequency 10 \
--num_agents 3 --critic_version 2 # --save_params

# eps0.6 5 agents
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_5agents_eps0.6 \
--critic_epsilon 0.6 --advice_dim 4
--seed 44 --actor_update_frequency 10 \
--num_agents 5 --critic_version 2 # --save_params

# eps1 2 agents
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_2agents_eps1 \
--critic_epsilon 1 --advice_dim 4
--seed 42 --actor_update_frequency 10 \
--num_agents 2 --critic_version 2 # --save_params

# eps1 3 agents
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_3agents_eps1 \
--critic_epsilon 1 --advice_dim 4
--seed 43 --actor_update_frequency 10 \
--num_agents 3 --critic_version 2 # --save_params

# eps1 5 agents
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 10 \
--n_iter 1000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_v2_5agents_eps1 \
--critic_epsilon 1 --advice_dim 4
--seed 44 --actor_update_frequency 10 \
--num_agents 5 --critic_version 2 # --save_params