python cs285/scripts/run_sac_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
-n 2000000 -l 2 -s 256 -b 1500 -eb 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_hw3q6_hprm_3agents \
--seed 41 --actor_update_frequency 10 --critic_epsilon 0.3 \
--num_agents 3 --advice_dim 3