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

### Merge equilibriums
eq_dir_prefix = os.path.join(dir_prefix,'equilibriums')

## nature
# before
fname_prefix = 'nature_eq_before'
nature_eq_fname = fname_prefix + file_suffix

dfs = []
files_not_found = {fname_prefix:[]}
for s in range(seed_lb, seed_ub):
	fname = os.path.join(eq_dir_prefix, nature_eq_fname)
	fname = fname % s
	try:
		df = pd.read_csv(fname)
		dfs.append(df)
	except FileNotFoundError:
		print('couldnt find',fname)
		files_not_found[fname_prefix].append(s)

merged_nature_eq_fname = fname_prefix + merged_file_suffix
merged_nature_eq_fname = os.path.join(eq_dir_prefix, merged_nature_eq_fname)
nature_eq_dfs = pd.concat(dfs)
print(merged_nature_eq_fname)
nature_eq_dfs.to_csv(merged_nature_eq_fname, index=False)



# after
fname_prefix = 'nature_eq_after'
nature_eq_fname = fname_prefix + file_suffix

dfs = []
files_not_found = {fname_prefix:[]}
for s in range(seed_lb, seed_ub):
	fname = os.path.join(eq_dir_prefix, nature_eq_fname)
	fname = fname % s
	try:
		df = pd.read_csv(fname)
		dfs.append(df)
	except FileNotFoundError:
		files_not_found[fname_prefix].append(s)

merged_nature_eq_fname = fname_prefix + merged_file_suffix
merged_nature_eq_fname = os.path.join(eq_dir_prefix, merged_nature_eq_fname)
nature_eq_dfs = pd.concat(dfs)
print(merged_nature_eq_fname)
nature_eq_dfs.to_csv(merged_nature_eq_fname, index=False)





## agent
# before
fname_prefix = 'agent_eq_before'
agent_eq_fname = fname_prefix + file_suffix

dfs = []
files_not_found[fname_prefix] = []
for s in range(seed_lb, seed_ub):
	fname = os.path.join(eq_dir_prefix, agent_eq_fname)
	fname = fname % s
	try:
		df = pd.read_csv(fname)
		dfs.append(df)
	except FileNotFoundError:
		files_not_found[fname_prefix].append(s)

merged_agent_eq_fname = fname_prefix + merged_file_suffix
merged_agent_eq_fname = os.path.join(eq_dir_prefix, merged_agent_eq_fname)
agent_eq_dfs = pd.concat(dfs)
print(merged_agent_eq_fname)
agent_eq_dfs.to_csv(merged_agent_eq_fname, index=False)


# after
fname_prefix = 'agent_eq_after'
agent_eq_fname = fname_prefix + file_suffix

dfs = []
files_not_found[fname_prefix] = []
for s in range(seed_lb, seed_ub):
	fname = os.path.join(eq_dir_prefix, agent_eq_fname)
	fname = fname % s
	try:
		df = pd.read_csv(fname)
		dfs.append(df)
	except FileNotFoundError:
		files_not_found[fname_prefix].append(s)

merged_agent_eq_fname = fname_prefix + merged_file_suffix
merged_agent_eq_fname = os.path.join(eq_dir_prefix, merged_agent_eq_fname)
agent_eq_dfs = pd.concat(dfs)
print(merged_agent_eq_fname)
agent_eq_dfs.to_csv(merged_agent_eq_fname, index=False)


### Merge regrets
regret_dir_prefix = os.path.join(dir_prefix,'regrets')

fname_prefix = 'regret'
regret_fname = fname_prefix + file_suffix

dfs = []
files_not_found[fname_prefix] = []
for s in range(seed_lb, seed_ub):
	fname = os.path.join(regret_dir_prefix, regret_fname)
	fname = fname % s
	try:
		df = pd.read_csv(fname)
		dfs.append(df)
	except FileNotFoundError:
		files_not_found[fname_prefix].append(s)

merged_regret_fname = fname_prefix + merged_file_suffix
merged_regret_fname = os.path.join(regret_dir_prefix, merged_regret_fname)
regret_dfs = pd.concat(dfs)
print(merged_regret_fname)
regret_dfs.to_csv(merged_regret_fname, index=False)

print('Files not found:')
print(files_not_found)


