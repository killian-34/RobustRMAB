import numpy as np
import pandas as pd
import sys, os
import matplotlib.pyplot as plt

#Ensure type 1 fonts are used
import matplotlib as mpl
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.unicode']=True


SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
plt.rc('font', weight='bold')
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.figure()

# e.g., armman_v1
batch_name = sys.argv[1]
robust_keyword = sys.argv[2]
data_name = sys.argv[3]

tick_names = ['No Action','Random','Hawkins','RMABPPO']
colors=['#000000', '#ff6b6b', '#406cff', '#9c1010']

# e.g., _n5_b1.0_h10_epoch5_dataarmman_
# file_token = sys.argv[2]

N_list = []
B_list= [] # goes with N
S = 10
ftoken = "_%s_n%s_b%s_h%s_data%s_r%s_p%s_"
hatch_list = [None, None, '/', '.']
h=10 # horizon length

out_name = "final_loop"

# data_name = "counterexample"
# data_name = "armman"
# data_name = "sis"
if data_name == 'counterexample':
	N_list = [3, 21, 48]
	B_list= [1.0, 7.0, 16.0] # goes with N
	S=10
elif data_name == 'armman':
	N_list = [5, 25, 50]
	B_list = [1.0, 5.0, 10.0]
	S=10
elif data_name == 'sis':
	N_list = [5, 10, 15]
	B_list = [4.0, 8.0, 12.0]
	S=50





fig, ax = None, None
if data_name == 'sis':
	fig, ax = plt.subplots(1,2, figsize=(16,2.3), gridspec_kw={'width_ratios': [3, 5]})
else:
	fig, ax = plt.subplots(1,2, figsize=(16,2.3), gridspec_kw={'width_ratios': [3, 3]})





plot_data = {}
dir_prefix = os.path.join('batches', batch_name)
results_dir_prefix = os.path.join(dir_prefix,'results')
fname_prefix = 'rewards'



######## ONE
# B=B_list[0]

for i,N in enumerate(N_list):

	B = B_list[i]
	filled_token = ftoken%(batch_name,N,B,h,data_name,robust_keyword, S)

	
	merged_file_suffix = filled_token+'merged.csv'

	### Merge regrets
	
	merged_rewards_fname = fname_prefix + merged_file_suffix
	merged_rewards_fname = os.path.join(results_dir_prefix, merged_rewards_fname)

	rewards_df = pd.read_csv(merged_rewards_fname)


	data = rewards_df.to_numpy() / N

	n_measurements = data.shape[0]
	n_methods = data.shape[1]
	means = data.mean(axis=0) 
	sem = data.std(axis=0)/np.sqrt(n_measurements)*1.96

	######## TWO

	plot_data[N] = {
						'n_measurements':n_measurements, 
						'n_methods':n_methods,
						'means':means,
						'sem':sem
					}



######## THREE
n_measurements = np.array([plot_data[n]['n_measurements'] for n in N_list ])
n_methods = np.array([plot_data[n]['n_methods'] for n in N_list ])[0]

print(tick_names)
print(plot_data[N])



x = np.arange(len(N_list))


######## FOUR
# for i,N in enumerate(N_list):

bar_width=0.2
for i in range(n_methods):
	ax[0].bar(x+bar_width*i, [plot_data[j]['means'][i] for j in N_list], 
		width=bar_width,
		yerr=[plot_data[j]['sem'][i] for j in N_list], 
		color=colors[i], 
		label=tick_names[i], 
		edgecolor='black',
		hatch=hatch_list[i])
	ax[0].set_xticks(x+bar_width*(n_methods-1)/2)
	ax[0].set_xticklabels(['N:%i,B:%i'%(N_list[i], B_list[i]) for i in range(len(N_list))])
all_data= np.array([plot_data[N]['means'] for N in N_list])

y_min, y_max = all_data.min(), all_data.max()

y_range = y_max - y_min

adjust = 0.05
y_min, y_max = y_min-adjust*y_range, y_max + adjust*y_range

ax[0].set_ylim([y_min, y_max])


####################################################################

############################

####################

N=0
B=0
S_list = []
if data_name == 'counterexample':
	N = 21
	B_list = [4.0, 7.0, 10.0]
elif data_name == 'armman':
	N = 25
	B_list = [3.0, 5.0, 7.0]
elif data_name == 'sis':
	N = 5
	B = 4.0
	S_list = [50, 100, 200]#, 500, 1000]

h=10 # horizon length


plot_data = {}
dir_prefix = os.path.join('batches', batch_name)
results_dir_prefix = os.path.join(dir_prefix,'results')
fname_prefix = 'rewards'



if data_name == 'sis':
	for S in S_list:
		# B = B_list[i]
		filled_token = ftoken%(batch_name,N,B,h,data_name,robust_keyword, S)

		
		merged_file_suffix = filled_token+'merged.csv'

		### Merge regrets
		
		merged_rewards_fname = fname_prefix + merged_file_suffix
		merged_rewards_fname = os.path.join(results_dir_prefix, merged_rewards_fname)

		rewards_df = pd.read_csv(merged_rewards_fname)

		# tick_names = rewards_df.columns.values

		data = rewards_df.to_numpy() / N

		# we ran No Action for Hawkins as a placeholder for the scripts --
		# now replace those values with 0 in the plot so it doesn't give the 
		# impression that performance dropped off -- it just wasn't run
		if S > 200:
			data[:,2] = 0 # zero out the hawkins runs that aren't real


		n_measurements = data.shape[0]
		n_methods = data.shape[1]
		means = data.mean(axis=0) 
		sem = data.std(axis=0)/np.sqrt(n_measurements)*1.96

		######## TWO

		# plot_data[N] = {
		plot_data[S] = {
							'n_measurements':n_measurements, 
							'n_methods':n_methods,
							'means':means,
							'sem':sem
						}



	######## THREE
	n_measurements = np.array([plot_data[S]['n_measurements'] for S in S_list ])
	n_methods = np.array([plot_data[S]['n_methods'] for S in S_list ])[0]


	print(tick_names)
	print(means)

	# tick_names = ['No Action','Random','Hawkins','RMABPPO']

	x = np.arange(len(S_list))
	# fig, ax = plt.subplots(1, len(B_list), figsize=(10,4))
	# ax = ax.reshape(-1)

	# colors=['#002222', '#335577', '#5599cc', '#bbddff', '#ddeeff']


	######## FOUR
	# for i,N in enumerate(N_list):

	bar_width=0.2
	for i in range(n_methods):
		ax[1].bar(x+bar_width*i, [plot_data[j]['means'][i] for j in S_list], 
			width=bar_width,
			yerr=[plot_data[j]['sem'][i] for j in S_list], 
			color=colors[i], 
			label=tick_names[i], 
			edgecolor='black',
			hatch=hatch_list[i])
	# for i,B in enumerate(B_list):
	# 	ax[i].bar(x, plot_data[B]['means'], yerr=plot_data[B]['std'])
		ax[1].set_xticks(x+bar_width*(n_methods-1)/2)
		ax[1].set_xticklabels(['S:%i'%(S_list[i]) for i in range(len(S_list))])

	all_data= np.array([plot_data[N]['means'] for N in S_list])
	# ax[0].set_ylim([y_min, y_max])
	ax[1].set_ylim([y_min, y_max])

else:
	for B in B_list:
		# B = B_list[i]
		filled_token = ftoken%(batch_name,N,B,h,data_name,robust_keyword, S)

		
		merged_file_suffix = filled_token+'merged.csv'

		### Merge regrets
		
		merged_rewards_fname = fname_prefix + merged_file_suffix
		merged_rewards_fname = os.path.join(results_dir_prefix, merged_rewards_fname)

		rewards_df = pd.read_csv(merged_rewards_fname)

		# tick_names = rewards_df.columns.values

		data = rewards_df.to_numpy() / N

		n_measurements = data.shape[0]
		n_methods = data.shape[1]
		means = data.mean(axis=0) 
		sem = data.std(axis=0)/np.sqrt(n_measurements)*1.96

		######## TWO

		# plot_data[N] = {
		plot_data[B] = {
							'n_measurements':n_measurements, 
							'n_methods':n_methods,
							'means':means,
							'sem':sem
						}



	######## THREE
	# n_measurements = np.array([plot_data[n]['n_measurements'] for n in N_list ])
	# n_methods = np.array([plot_data[n]['n_methods'] for n in N_list ])[0]
	n_measurements = np.array([plot_data[B]['n_measurements'] for B in B_list ])
	n_methods = np.array([plot_data[B]['n_methods'] for B in B_list ])[0]

	print(tick_names)
	print(means)

	# tick_names = ['No Action','Random','Hawkins','RMABPPO']

	x = np.arange(len(B_list))
	# fig, ax = plt.subplots(1, len(B_list), figsize=(10,4))
	# ax = ax.reshape(-1)

	# colors=['#002222', '#335577', '#5599cc', '#bbddff', '#ddeeff']


	######## FOUR
	# for i,N in enumerate(N_list):

	bar_width=0.2
	for i in range(n_methods):
		ax[1].bar(x+bar_width*i, [plot_data[j]['means'][i] for j in B_list], 
			width=bar_width,
			yerr=[plot_data[j]['sem'][i] for j in B_list], 
			color=colors[i], 
			label=tick_names[i], 
			edgecolor='black',
			hatch=hatch_list[i])
	# for i,B in enumerate(B_list):
	# 	ax[i].bar(x, plot_data[B]['means'], yerr=plot_data[B]['std'])
		ax[1].set_xticks(x+bar_width*(n_methods-1)/2)
		ax[1].set_xticklabels(['N:%i,B:%i'%(N, B_list[i]) for i in range(len(B_list))])

	all_data= np.array([plot_data[N]['means'] for N in B_list])

	


y_min2, y_max2 = all_data.min(), all_data.max()
# y_min, y_max = min(plot_data[B]['means']), max(plot_data[B]['means'])
# y_range = y_max - y_min

# adjust = 0.05
# y_min, y_max = y_min-adjust*y_range, y_max + adjust*y_range

y_min = min(y_min, y_min2)
y_max = max(y_max, y_max2)

y_range = y_max - y_min

adjust = 0.05
y_min, y_max = y_min-adjust*y_range, y_max + adjust*y_range

if data_name == 'counterexample' or data_name == 'armman':
	ax[0].set_ylim([y_min, y_max])
	ax[1].set_ylim([y_min, y_max])



####################################################################################

grey_bg = "#f2f2f2"
ax[0].grid(zorder=0)
ax[1].grid(zorder=0)
ax[0].set_facecolor(grey_bg)
ax[1].set_facecolor(grey_bg)


ax[0].set_ylabel('Reward/N')
if data_name == 'sis':
	pass
	# ax[0].legend(loc='upper center', bbox_to_anchor=[1, 0.3, 1, 1], ncol=4)
elif data_name == 'counterexample':
	ax[0].legend(loc='upper center', bbox_to_anchor=[0.5, 0.55, 1, 1], ncol=4)
# ax[0].legend(loc='upper center', bbox_to_anchor=[0.5, 0.5, 1, 1], ncol=4)

####### SIX
# fig.suptitle('N:{}, B:{}, H:{}, Data: {}, Nature: {}, N_sims:{}'.format(N_list, B_list, h, data_name, robust_keyword, n_measurements))
# fig.suptitle('N:{}, B:{}, H:{}, Data: {}, Nature: {}, N_sims:{}'.format(N, B_list, h, data_name, robust_keyword, n_measurements))
# fig.tight_layout(rect=[0,0,.8,1])
plt.subplots_adjust(bottom=0.2,top=0.7)


####### SEVEN
plt.savefig(os.path.join('.', 'img/final/rewards_{}_h{}_data{}_r{}.pdf'.format(out_name, h, data_name, robust_keyword)))
# plt.savefig(os.path.join('.', 'img/rewards_{}_n{}_h{}_data{}_r{}.pdf'.format(out_name, N, h, data_name, robust_keyword)))
plt.show()





#