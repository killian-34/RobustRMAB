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
fig, ax = None, None

out_name = "final_loop_appendix"

# ax.grid()

# e.g., armman_v1
batch_name = sys.argv[1]
data_name = sys.argv[2]

# ['Double Oracle', 'Double Oracle+New Nats', 'RL vs. Middle',
#        'Random Agent', 'Hawkins Pessimist Agent', 'Hawkins Middle Agent',
#        'Hawkins Optimist Agent'
# ]
DROP_COLS = ['Double Oracle+New Nats']

# e.g., _n5_b1.0_h10_epoch5_dataarmman_
# file_token = sys.argv[2]

ftoken = "_%s_n%s_b%s_h%s_epoch%s_data%s_"


tick_names = ['RMABDO', 'RLvMid', 'Rand', 'HP', 'HM', 'HO']
colors=['#9c1010', '#ff6b6b', '#000000', '#406cff', '#92c7fc', '#e0f0ff']
hatch_list = [None, None, None, '/', '/', '/']

N_list = None
B = 0
S_list = []
B_list = []

if data_name == "counterexample":
	N_list = [21, 33, 48]
	B = 1.0
	tick_names = ['RMABDO', 'RLvMid', 'Rand', 'HP', 'HM', 'HO']
	colors=['#9c1010', '#ff6b6b', '#000000', '#406cff', '#92c7fc', '#e0f0ff']
	hatch_list = [None, None, None, '/', '/', '/']
	fig, ax = plt.subplots(1,1, figsize=(8,2.3))
	ftoken = "_%s_n%s_b%s_h%s_epoch%s_data%s_p10_"
	ax = np.array([ax])

elif data_name == 'armman':
	N_list = [20, 25, 35]
	B = 1.0
	fig, ax = plt.subplots(1,1, figsize=(8,2.3))
	ftoken = "_%s_n%s_b%s_h%s_epoch%s_data%s_p10_"
	ax = np.array([ax])
	# DROP_COLS = ['Double Oracle+New Nats','Random Agent']
	# tick_names = ['DO', 'RLvMid', 'HP', 'HM', 'HO']
	# colors=['#9c1010', '#ff6b6b', '#406cff', '#92c7fc', '#e0f0ff']
	# hatch_list = [None, None, '/', '/', '/']
elif data_name == "sis":
	# data_name = "sis"
	batch_name = 'do_sis_final_wbaselines_appendix_armscale'
	N_list = [12]
	# N_list = [8, 10, 12, 15]
	
	B_list = [10.0]
	# B_list = [6.0, 8.0, 10.0, 12.0]

	S=50
	ftoken = "_%s_n%s_b%s_h%s_epoch%s_data%s_p%s_"
	fig, ax = plt.subplots(1,2, figsize=(12,2.3), gridspec_kw={'width_ratios': [1, 2]})
	ax.reshape(-1)




h=10 # horizon length
epoch=5 # num double oracle loops
# epoch=7 # num double oracle loops



plot_data = {}
dir_prefix = os.path.join('batches', batch_name)
regret_dir_prefix = os.path.join(dir_prefix,'regrets')
fname_prefix = 'regret'



######## ONE
# B=B_list[0]

for i,N in enumerate(N_list):
# out_name = "b_loop_%s"%data_name
# N=N_list[1]
# for B in B_list:
	filled_token = None
	if data_name == 'sis':
		B = B_list[i]
		filled_token = ftoken%(batch_name,N,B,h,epoch,data_name,S)
	else:
		filled_token = ftoken%(batch_name,N,B,h,epoch,data_name)

	
	merged_file_suffix = filled_token+'merged.csv'

	### Merge regrets
	
	merged_regret_fname = fname_prefix + merged_file_suffix
	merged_regret_fname = os.path.join(regret_dir_prefix, merged_regret_fname)

	regret_df = pd.read_csv(merged_regret_fname)
	regret_df.drop(columns=DROP_COLS, inplace=True)

	# tick_names = regret_df.columns.values

	data = regret_df.to_numpy()

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
print(plot_data)

# tick_names = ['No Action','Random','Hawkins','RMABPPO']

x = np.arange(len(N_list))
# fig, ax = plt.subplots(1, len(B_list), figsize=(10,4))
# ax = ax.reshape(-1)




######## FOUR
# for i,N in enumerate(N_list):

bar_width=0.12
for i in range(n_methods):
	ax[0].bar(x+bar_width*i, [plot_data[j]['means'][i] for j in N_list], 
		width=bar_width,
		yerr=[plot_data[j]['sem'][i] for j in N_list], 
		color=colors[i], 
		label=tick_names[i], 
		edgecolor='black',
		hatch=hatch_list[i],
		zorder=1)
# for i,B in enumerate(B_list):
# 	ax[i].bar(x, plot_data[B]['means'], yerr=plot_data[B]['sem'])

	ax[0].set_xticks(x+bar_width*(n_methods-1)/2)
	if data_name == 'sis':
		ax[0].set_xticklabels(['N:%i,B:%i'%(N_list[i], B_list[i]) for i in range(len(N_list))])
	else:
		ax[0].set_xticklabels(['N:%i,B:%i'%(N_list[i], B) for i in range(len(N_list))])
all_data= np.array([plot_data[N]['means'] for N in N_list])

y_min, y_max = all_data.min(), all_data.max()
# y_min, y_max = min(plot_data[B]['means']), max(plot_data[B]['means'])
y_range = y_max - y_min

adjust = 0.05
y_min, y_max = y_min-adjust*y_range, y_max + adjust*y_range

ax[0].set_ylim([y_min, y_max])

top_tick = round(y_max-adjust*y_range)
yticks = np.round(np.linspace(0, top_tick, 4),1)
ax[0].set_yticks(yticks)


if data_name == 'armman' or data_name == 'counterexample':
	grey_bg = "#f2f2f2"
	ax[0].grid(zorder=0)
	ax[0].set_facecolor(grey_bg)
	ax[0].set_ylabel('Regret/N')
	# ax[0].legend(loc='upper center', bbox_to_anchor=[1.0, 0, 1, 1], ncol=2)
	# ax[0].legend(loc='upper center', bbox_to_anchor=[0.5, 0.5, 1, 1], ncol=4)
	plt.subplots_adjust(bottom=0.2, left=0.2, right=0.8, top=0.8 )
	plt.savefig(os.path.join('.', 'img/final/regret_{}_h{}_epoch{}_data{}.pdf'.format(out_name, h, epoch, data_name)))
	plt.show()
	exit()





####################################################################

############################

####################

# counterexample
if data_name == 'sis':
	B=4.0
	N=5
	S_list = [100, 200, 500]
	batch_name = 'do_sis_final_wbaselines_appendix_statescale'

	# Can't run Hawkins for larger state spaces but keep the bars for now
	tick_names = ['RMABDO', 'RLvMid', 'Rand']#, 'HP', 'HM', 'HO']
	# colors=['#9c1010', '#ff6b6b', '#000000', '#406cff', '#92c7fc', '#e0f0ff']
	# hatch_list = [None, None, None, '/', '/', '/']

h=10 # horizon length



plot_data = {}
dir_prefix = os.path.join('batches', batch_name)
regret_dir_prefix = os.path.join(dir_prefix,'regrets')
fname_prefix = 'regret'

for S in S_list:

	filled_token = None
	
	filled_token = ftoken%(batch_name,N,B,h,epoch,data_name,S)

	
	merged_file_suffix = filled_token+'merged.csv'

	### Merge regrets
	
	merged_regret_fname = fname_prefix + merged_file_suffix
	merged_regret_fname = os.path.join(regret_dir_prefix, merged_regret_fname)

	regret_df = pd.read_csv(merged_regret_fname)
	regret_df.drop(columns=DROP_COLS, inplace=True)

	# tick_names = regret_df.columns.values

	data = regret_df.to_numpy()

	n_measurements = data.shape[0]
	n_methods = data.shape[1]
	means = data.mean(axis=0)
	sem = data.std(axis=0)/np.sqrt(n_measurements)*1.96

	######## TWO

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


n_methods = 3 # don't plot hawkins for larger state spaces (couldn't run in time)

x = np.arange(len(S_list))


######## FOUR
# for i,N in enumerate(N_list):

bar_width=0.24
for i in range(n_methods):
	ax[1].bar(x+bar_width*i, [plot_data[j]['means'][i] for j in S_list], 
		width=bar_width,
		yerr=[plot_data[j]['sem'][i] for j in S_list], 
		color=colors[i], 
		label=tick_names[i], 
		edgecolor='black',
		hatch=hatch_list[i],
		zorder=1)
# for i,B in enumerate(B_list):
# 	ax[i].bar(x, plot_data[B]['means'], yerr=plot_data[B]['sem'])
	ax[1].set_xticks(x+bar_width*(n_methods-1)/2)
	ax[1].set_xticklabels(['S:%i'%(S_list[i]) for i in range(len(S_list))])

all_data= np.array([plot_data[S]['means'] for S in S_list])

y_min2, y_max2 = all_data.min(), all_data.max()
# y_min2, y_max2 = min(plot_data[B]['means']), max(plot_data[B]['means'])
y_range2 = y_max2 - y_min2

adjust = 0.05
y_min2, y_max2 = y_min2-adjust*y_range2, y_max2 + adjust*y_range2
ax[1].set_ylim([y_min2, y_max2])

top_tick = round(y_max2-adjust*y_range2)
yticks = np.round(np.linspace(0, top_tick, 4),1)
ax[1].set_yticks(yticks)

# Do this to make both y-axes the same
y_min = min(y_min, y_min2)
y_max = max(y_max, y_max2)

y_range = y_max - y_min

adjust = 0.05
y_min, y_max = y_min-adjust*y_range, y_max + adjust*y_range


# uncomment to equalize both y scales
# ax[0].set_ylim([y_min, y_max])
# ax[1].set_ylim([y_min, y_max])
# top_tick = round(y_max-adjust*y_range)
# yticks = np.round(np.linspace(0, top_tick, 5),1)
# ax[0].set_yticks(yticks)
# ax[1].set_yticks(yticks)

# uncomment to just set both at 5
# if data_name == 'armman':
# 	y_min = 0 
# 	y_max = 7
# 	ax[0].set_ylim([y_min, y_max+0.3])
# 	ax[1].set_ylim([y_min, y_max+0.3])
# 	top_tick = round(y_max)
# 	yticks = np.round(np.linspace(0, top_tick, 5),1)
# 	ax[0].set_yticks(yticks)
# 	ax[1].set_yticks(yticks)


####################################################################################

grey_bg = "#f2f2f2"
ax[0].grid(zorder=0)
ax[1].grid(zorder=0)
ax[0].set_facecolor(grey_bg)
ax[1].set_facecolor(grey_bg)





ax[0].set_ylabel('Regret/N')
if data_name == 'sis':
	ax[0].legend(loc='lower center', bbox_to_anchor=[1.0, -0.7, 1, 1], ncol=6)
# ax[0].legend(loc='upper center', bbox_to_anchor=[0.5, 0.5, 1, 1], ncol=4)

####### SIX
# fig.suptitle('N:{}, B:{}, H:{}, Data: {}, Nature: {}, N_sims:{}'.format(N_list, B_list, h, data_name, robust_keyword, n_measurements))
# fig.suptitle('N:{}, B:{}, H:{}, Data: {}, Nature: {}, N_sims:{}'.format(N, B_list, h, data_name, robust_keyword, n_measurements))
# fig.tight_layout(rect=[0,0,.8,1])
plt.subplots_adjust(bottom=0.4,top=0.98)


####### SEVEN
plt.savefig(os.path.join('.', 'img/final/regret_{}_h{}_epoch{}_data{}.pdf'.format(out_name, h, epoch, data_name)))

plt.show()





#