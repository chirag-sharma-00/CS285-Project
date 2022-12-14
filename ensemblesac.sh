python cs285/scripts/run_sac_ensemble_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
--n_iter 100000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_ensemble_2_agents_100k_iter \
--seed 6 --actor_update_frequency 10 \
--num_agents 2 & # --save_params

python cs285/scripts/run_sac_ensemble_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
--n_iter 100000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_ensemble_3_agents_100k_iter \
--seed 6 --actor_update_frequency 10 \
--num_agents 3 & # --save_params

python cs285/scripts/run_sac_ensemble_experiment.py \
--env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.99 --scalar_log_freq 1500 \
--n_iter 100000 --n_layers 2 --size 256 --batch_size 1500 --eval_batch_size 1500 \
-lr 0.0003 --init_temperature 0.1 --exp_name sac_ensemble_5_agents_100k_iter \
--seed 6 --actor_update_frequency 10 \
--num_agents 5 # --save_params