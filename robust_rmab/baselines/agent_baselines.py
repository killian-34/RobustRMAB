import numpy as np
import itertools



class PessimisticAgentPolicy:
    def __init__(self, N, ind):
        self.N=N
        self.ind = ind
        self.name="Pessimist_Agent"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def act_test(self, o):
        
        # Return 0 for all arms
        return np.zeros(self.N)


class RandomAgentPolicy:
    def __init__(self, env, ind):
        self.env = env

        self.ind = ind
        self.name="Random_Agent"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def act_test(self, o):
        return self.env.random_agent_action()


class HawkinsAgentPolicy:
    def __init__(self, N, T, R, C, B, gamma, ind):
        from robust_rmab import lp_methods
        self.lp_methods = lp_methods
        self.N = N
        self.T = T
        self.R = R
        self.C = C
        self.B = B
        self.gamma = gamma

        self.ind = ind
        self.name="Hawkins"

    def __repr__(self):
        return "%s_%s"%(self.name, self.ind)

    def act_test(self, o, just_hawkins_lambda=False):

        
        current_state = o
        if type(o) != np.ndarray:
            current_state = o.numpy()

        actions = np.zeros(self.N)

        lambda_lim = self.R.max()/(self.C[self.C>0].min()*(1-self.gamma))

        indexes = np.zeros((self.N, self.C.shape[0], self.T.shape[1]))
        current_state = current_state.reshape(-1)
        current_state = current_state.astype(int)
        L_vals, lambda_val, obj_val = self.lp_methods.hawkins(self.T, self.R, self.C, self.B, current_state, lambda_lim=lambda_lim, gamma=self.gamma)


        for i in range(self.N):
            for a in range(self.C.shape[0]):
                for s in range(self.T.shape[1]):
                    indexes[i,a,s] = self.R[i,s] - lambda_val*self.C[a] + self.gamma*L_vals[i].dot(self.T[i,s,a])
        # data_dict['hawkins_lambda'].append(lambda_val)
        if just_hawkins_lambda:
            print('state', current_state)
            print('L_vals', L_vals)
            print('lambda',lambda_val)
            print('obj_val',obj_val)
            1/0

        indexes_per_state = np.zeros((self.N, self.C.shape[0]))
        for i in range(self.N):
            s = current_state[i]
            # print(s)
            indexes_per_state[i] = indexes[i,:,s]

        # start = time.time()

        decision_matrix = self.lp_methods.action_knapsack(indexes_per_state, self.C, self.B)

        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")

        payment = 0
        for i in range(len(actions)):
            payment += self.C[actions[i]]
        if not payment <= self.B:
            print("budget")
            print(self.B)
            print("Cost")
            print(self.C)
            print("ACTIONS")
            print(actions)
            raise ValueError("Over budget")

        
        # print("T:",self.T)
        # print("state",current_state)
        # print("actions:",actions)
        # print("Hawkins:")
        # print("ind:",self.ind)
        # print()


        return actions


    def get_policy_array(self):
        import itertools

        all_states = list(itertools.product(np.arange(self.T.shape[1]), repeat=self.T.shape[0]))
        policy_array = np.zeros(len(all_states),dtype=int)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        for i in range(all_states.shape[0]):
            actions = self.act_test(all_states[i])
            action_ind = np.argmax(actions)
            policy_array[i] = action_ind

        return policy_array, tup_to_ind

        






