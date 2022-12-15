# Use epsilon = 0.3
seed=42
for i in 3 5
do
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
-n 100000 -l 2 -s 256 -b 1500 -eb 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersac_$i\agents_eps1 \
--seed $seed --actor_update_frequency 10 \
--critic_epsilon 1 --advice_dim 4 --num_agents $i --critic_version 2 

seed=$((seed+1))
done

