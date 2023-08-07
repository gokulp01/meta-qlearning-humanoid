import argparse
import os

def humanoid_dir(seed, log_id, gpu_id):

	cmd = " python models/run.py --env_name humanoid-dir \
	--alg_name mql --policy_freq 3 --expl_noise 0.2 \
	--enable_context  --num_train_steps 5000000 --cuda_deterministic \
	--history_length  150  --unbounded_eval_hist  --beta_clip 2 \
	--enable_adaptation  --num_initial_steps 1000 --main_snap_iter_nums 15 \
	--snap_iter_nums 5 --hidden_sizes  300 300  --lam_csc  0.1 --snapshot_size 2000 \
	--lr  0.0003  --sample_mult 5  --use_epi_len_steps  --enable_beta_obs_cxt \
	--hiddens_conext 20   --num_tasks_sample 1 --burn_in  10000 --batch_size 256 \
	--policy_noise 0.3 --eval_freq 10000  --replay_size 500000  "+ \
	' --log_id ' + log_id + ' --seed ' + str(seed) + ' --gpu_id ' + str(gpu_id)
	return cmd


if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='ant-goal')
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--gpu_id', type=int, default=0)
	parser.add_argument('--log_id', type=str, default='dummy')
	args = parser.parse_args()
	print('------------')
	print(args.__dict__)
	print('------------')

	cmd = ''

	cmd = humanoid_dir(args.seed, args.log_id, args.gpu_id)


	print(cmd)
	os.system(cmd)










