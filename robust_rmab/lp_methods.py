from gurobipy import *
import numpy as np 
import sys
import time





# https://dspace.mit.edu/handle/1721.1/29599
def hawkins(T, R, C, B, start_state, lambda_lim=None, gamma=0.95):

	start = time.time()

	NPROCS = T.shape[0]
	NSTATES = T.shape[1]
	NACTIONS = T.shape[2]

	# Create a new model
	m = Model("LP for Hawkins Lagrangian relaxation")
	m.setParam( 'OutputFlag', False )

	L = np.zeros((NPROCS,NSTATES),dtype=object)
	
	mu = np.zeros((NPROCS,NSTATES),dtype=object)
	for i in range(NPROCS):
		# mu[i] = np.random.dirichlet(np.ones(NSTATES))
		mu[i, int(start_state[i])] = 1

	c = C

	# Create variables
	lb = 0
	ub = GRB.INFINITY
	if lambda_lim is not None:
		ub = lambda_lim

	index_variable = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='index')


	for p in range(NPROCS):
		for i in range(NSTATES):
			L[p,i] = m.addVar(vtype=GRB.CONTINUOUS, name='L_%s_%s'%(p,i))


	L = np.array(L)


	# print('Variables added in %ss:'%(time.time() - start))
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first
	# minimze the value function

	# In Hawkins, only min the value function of the start state
	# print(current_state)
	# m.setObjectiveN(sum([L[i][current_state[i]] for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)
	tiny=1e-6
	m.setObjectiveN(sum([L[i].dot(mu[i]) for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1) + tiny*L.sum(), 0, 1)

	# set constraints
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				# m.addConstr( L[p][i] >= R[p][i] - index_variable*c[j] + gamma*L[p].dot(T[p,i,j]) )
				m.addConstr( L[p][i] >= R[p][i] - index_variable*c[j] + gamma*LinExpr(T[p,i,j], L[p])) 



	# print('Constraints added in %ss:'%(time.time() - start))
	start = time.time()

	# Optimize model

	m.optimize()
	# m.printStats()

	# print('Model optimized in %ss:'%(time.time() - start))
	start = time.time()


	L_vals = np.zeros((NPROCS,NSTATES))

	index_solved_value = 0
	for v in m.getVars():
		if 'index' in v.varName:
			index_solved_value = v.x

		if 'L' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			L_vals[i,j] = v.x

	# print('Variables extracted in %ss:'%(time.time() - start))
	start = time.time()

	obj = m.getObjective()
	

	return L_vals, index_solved_value, obj.getValue()




# https://dspace.mit.edu/handle/1721.1/29599
def lp_to_compute_index(T, R, C, B, start_state, a_index, lambda_lim=None, gamma=0.95):

	start = time.time()

	NPROCS = T.shape[0]
	NSTATES = T.shape[1]
	NACTIONS = T.shape[2]

	# Create a new model
	m = Model("LP for Computing multi-action indices")
	m.setParam( 'OutputFlag', False )

	L = np.zeros((NPROCS,NSTATES),dtype=object)
	
	mu = np.zeros((NPROCS,NSTATES),dtype=object)
	for i in range(NPROCS):
		# mu[i] = np.random.dirichlet(np.ones(NSTATES))
		mu[i, int(start_state[i])] = 1

	c = C

	# Create variables
	lb = 0
	ub = GRB.INFINITY
	if lambda_lim is not None:
		ub = lambda_lim


	# going to compute indices in a decoupled manner
	index_variables = np.zeros(NPROCS,dtype=object)
	for i in range(NPROCS):
		index_variables[i] = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='index_%s'%i)


	for p in range(NPROCS):
		for i in range(NSTATES):
			L[p,i] = m.addVar(vtype=GRB.CONTINUOUS, name='L_%s_%s'%(p,i))


	L = np.array(L)


	# print('Variables added in %ss:'%(time.time() - start))
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first
	# minimze the value function

	# In Hawkins, only min the value function of the start state
	# print(current_state)
	# m.setObjectiveN(sum([L[i][current_state[i]] for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)

	m.setObjectiveN(sum([L[i].dot(mu[i]) for i in range(NPROCS)]) + index_variables[i]*B*((1-gamma)**-1), 0, 1)

	# set constraints
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				# m.addConstr( L[p][i] >= R[p][i] - index_variable*c[j] + gamma*L[p].dot(T[p,i,j]) )
				m.addConstr( L[p][i] >= R[p][i] - index_variables[p]*c[j] + gamma*LinExpr(T[p,i,j], L[p])) 


	# this computes the index
	# out of convenience it will assume actions are the same on all arms
	# and will compute them in parallel, even though arms are not coupled
	for p in range(NPROCS):
		m.addConstr(R[p][start_state[p]] - index_variables[p]*c[a_index] + gamma*LinExpr(T[p,start_state[p],a_index], L[p]) == R[p][start_state[p]] - index_variables[p]*c[a_index-1] + gamma*LinExpr(T[p,start_state[p],a_index-1], L[p]) ) 

	# print('Constraints added in %ss:'%(time.time() - start))
	start = time.time()

	# Optimize model

	m.optimize()
	# m.printStats()

	# print('Model optimized in %ss:'%(time.time() - start))
	start = time.time()


	L_vals = np.zeros((NPROCS,NSTATES))
	index_solved_values = np.zeros(NPROCS)

	for v in m.getVars():
		if 'index' in v.varName:
			i = int(v.varName.split('_')[1])
			index_solved_values[i] = v.x

		if 'L' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			L_vals[i,j] = v.x

	# print('Variables extracted in %ss:'%(time.time() - start))
	start = time.time()

	return L_vals, index_solved_values







# Transition matrix, reward vector, action cost vector
def action_knapsack(values, C, B):


	m = Model("Knapsack")
	m.setParam( 'OutputFlag', False )

	c = C

	x = np.zeros(values.shape, dtype=object)

	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i,j] = m.addVar(vtype=GRB.BINARY, name='x_%i_%i'%(i,j))



	m.modelSense=GRB.MAXIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first

	# minimze the value function
	m.setObjectiveN((x*values).sum(), 0, 1)

	# set constraints
	# m.addConstr( x.dot(C).sum() == B )
	m.addConstr( x.dot(C).sum() == B )
	for i in range(values.shape[0]):
		# m.addConstr( x[i].sum() <= 1 )
		m.addConstr( x[i].sum() == 1 )


	# Optimize model

	m.optimize()

	x_out = np.zeros(x.shape)

	for v in m.getVars():
		if 'x' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			x_out[i,j] = v.x

		else:
			pass
			# print((v.varName, v.x))

	# print(x_out)
	return x_out



