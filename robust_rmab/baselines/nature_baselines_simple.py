import numpy as np
import itertools


# Don't use if you need not deterministic (e.g., in main loop of double oracle)
class RandomNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.ind = ind
        self.name="Random_Nature"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        actions = np.zeros(self.nature_params.shape[0])
        
        for i in range(actions.shape[0]):
            param_range = self.nature_params[i][1] - self.nature_params[i][0]
            param_lower = self.nature_params[i][0]
            actions[i] = np.random.rand()*param_range + param_lower

        return actions

    def bound_nature_actions(self, actions):
        return actions

    # we'll just settle for getting one sample from each...
    def get_policy_array(self, state_dim=0, N=0):

        all_states = list(itertools.product(np.arange(state_dim), repeat=N))

        policy_array = np.zeros((len(all_states),N),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        for i in range(all_states.shape[0]):
            actions = self.get_nature_action(all_states[i])
            actions = self.bound_nature_actions(actions)
            policy_array[i] = actions

        return policy_array, tup_to_ind


class PessimisticNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.ind = ind
        self.name="Pessimist_Nature"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        
        # Return lower bound of all params
        return self.nature_params[:,0]

    def bound_nature_actions(self, actions):
        return actions

    def get_policy_array(self, state_dim=0, N=0):

        all_states = list(itertools.product(np.arange(state_dim), repeat=N))

        policy_array = np.zeros((len(all_states),N),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        # same for every state, so this is faster to compute
        policy_array[:] = self.get_nature_action(None)

        return policy_array, tup_to_ind

class OptimisticNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.ind = ind
        self.name="Optimist_Nature"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        
        # Return lower bound of all params
        return self.nature_params[:,1]

    def bound_nature_actions(self, actions):
        return actions

    def get_policy_array(self, state_dim=0, N=0):

        all_states = list(itertools.product(np.arange(state_dim), repeat=N))

        policy_array = np.zeros((len(all_states),N),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        # same for every state, so this is faster to compute
        policy_array[:] = self.get_nature_action(None)

        return policy_array, tup_to_ind


class MiddleNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.ind = ind
        self.name="Middle_Nature"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        
        # Return lower bound of all params
        return self.nature_params.mean(axis=1)

    def bound_nature_actions(self, actions):
        return actions


    def get_policy_array(self, state_dim=0, N=0):

        all_states = list(itertools.product(np.arange(state_dim), repeat=N))

        policy_array = np.zeros((len(all_states),N),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        # same for every state, so this is faster to compute
        policy_array[:] = self.get_nature_action(None)

        return policy_array, tup_to_ind