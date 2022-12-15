# 3 agents init_temp=0.2 eps=0.7
seed=92
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
-n 800000 -l 2 -s 256 -b 1500 -eb 1500 \
-lr 0.0003 --init_temperature 0.2 --exp_name peersac_bestparams \
--seed $seed --actor_update_frequency 10 \
--critic_epsilon 0.7 --advice_dim 2 --num_agents 2 --critic_version 2 

