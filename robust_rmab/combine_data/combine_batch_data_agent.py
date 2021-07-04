import pandas as pd
import sys, os

print('''
batch_name = sys.argv[1]

# e.g., _n5_b1.0_h10_epoch5_dataarmman_
file_token = sys.argv[2]

seed_ub = int(sys.argv[3])
	''')

# python3 combine_batch_data.py armman_v1 _n5_b1.0_h10_epoch5_dataarmman_ 49

# e.g., armman_v1
batch_name = sys.argv[1]

# e.g., _n5_b1.0_h10_epoch5_dataarmman_
file_token = sys.argv[2]

seed_lb = 0
seed_ub = int(sys.argv[3])


if len(sys.argv) > 4:
	seed_lb = int(sys.argv[3])
	seed_ub = int(sys.argv[4])

dir_prefix = os.path.join('batches', batch_name)
file_suffix = file_token + 's%i.csv'
merged_file_suffix = file_token+'merged.csv'

### Merge results
results_dir_prefix = os.path.join(dir_prefix,'results')

fname_prefix = 'rewards'
rewards_fname = fname_prefix + file_suffix

dfs = []
files_not_found = {fname_prefix:[]}
for s in range(seed_lb, seed_ub):
	fname = os.path.join(results_dir_prefix, rewards_fname)
	fname = fname % s
	try:
		df = pd.read_csv(fname)
		dfs.append(df)
	except FileNotFoundError:
		print('couldnt find',fname)
		files_not_found[fname_prefix].append(s)

merged_results_fname = fname_prefix + merged_file_suffix
merged_results_fname = os.path.join(results_dir_prefix, merged_results_fname)
results_dfs = pd.concat(dfs)
print(merged_results_fname)
results_dfs.to_csv(merged_results_fname, index=False)


print('Files not found:')
print(files_not_found)


