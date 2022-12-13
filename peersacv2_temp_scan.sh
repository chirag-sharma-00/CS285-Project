# 3 agents eps=0.3
seed = 42
for e in 0.05 0.1 0.2 0.4 0.8
do
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
-n 800000 -l 2 -s 256 -b 1500 -eb 1500 \
-lr 0.0003 --init_temperature $e --exp_name peersac_3agents_temp$e \
--seed $seed --actor_update_frequency 10 \
--critic_epsilon 0.3 --advice_dim 4 --num_agents 3 --critic_version 2 
seed=$((seed+1))
done

