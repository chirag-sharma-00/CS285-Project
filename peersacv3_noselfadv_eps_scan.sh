# Changes in v3: use peer advice in target. Fixed target copying rules. 
for e in 1 0.9 0.7 0.5 0.3 0.1
do
seed=$(date +%s)
python cs285/scripts/run_sac_peer_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
-n 800000 -l 2 -s 256 -b 1500 -eb 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name peersacv3_2agents_noselfadv_eps$e \
--seed $seed --actor_update_frequency 10 \
--critic_epsilon $e --advice_dim 4 --num_agents 2
done

