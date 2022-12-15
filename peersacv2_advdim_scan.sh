#!/bin/bash
seed=52
for i in 2 4 8
do
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
-n 400000 -l 2 -s 256 -b 1500 -eb 1500 \
-lr 0.0003 --init_temperature 0.2 --exp_name peersac_3agents_advdim$i \
--seed "$seed" --actor_update_frequency 10 \
--critic_epsilon 0.7 --advice_dim $i --num_agents 3 --critic_version 2 

seed=$((seed+1))
done

# Use epsilon = 0.7, agent_num=3, init_temp=0.2
