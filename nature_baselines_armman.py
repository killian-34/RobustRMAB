import numpy as np
import itertools


# Don't use if you need deterministic (e.g., in main loop of double oracle)
class RandomNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.ind = ind
        self.name="Random_Nature"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        actions = np.zeros((self.nature_params.shape[0],self.nature_params.shape[2]))
        
        for arm_i in range(actions.shape[0]):
            arm_state = int(o[arm_i])
            for arm_a in range(actions.shape[1]):
                param_range = self.nature_params[arm_i, arm_state, arm_a, 1] - self.nature_params[arm_i, arm_state, arm_a, 0]
                param_lower = self.nature_params[arm_i, arm_state, arm_a, 0]
                actions[arm_i, arm_a] = np.random.rand()*param_range + param_lower

        return actions

    def bound_nature_actions(self, actions, state=None, reshape=True):
        return actions

    # we'll just settle for getting one sample from each...
    def get_policy_array(self, state_dim=0, N=0):

        N = self.nature_params.shape[0]
        S = self.nature_params.shape[1]
        A = self.nature_params.shape[2]

        all_states = list(itertools.product(np.arange(S), repeat=N))

        policy_array = np.zeros((len(all_states),N*A),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        for i, state in enumerate(all_states):
            policy_array[i] = self.get_nature_action(state).reshape(-1)

        return policy_array, tup_to_ind


class PessimisticNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.ind = ind
        self.name="Pessimist_Nature"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        actions = np.zeros((self.nature_params.shape[0],self.nature_params.shape[2]))
        
        for arm_i in range(actions.shape[0]):
            arm_state = int(o[arm_i])
            for arm_a in range(actions.shape[1]):
                param_lower = self.nature_params[arm_i, arm_state, arm_a, 0]
                actions[arm_i, arm_a] = param_lower

        return actions

    def bound_nature_actions(self, actions, state=None, reshape=True):
        return actions

    def get_policy_array(self, state_dim=0, N=0):

        N = self.nature_params.shape[0]
        S = self.nature_params.shape[1]
        A = self.nature_params.shape[2]

        all_states = list(itertools.product(np.arange(S), repeat=N))

        policy_array = np.zeros((len(all_states),N*A),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        for i, state in enumerate(all_states):
            policy_array[i] = self.get_nature_action(state).reshape(-1)

        return policy_array, tup_to_ind

class OptimisticNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.ind = ind
        self.name="Optimist_Nature"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        actions = np.zeros((self.nature_params.shape[0],self.nature_params.shape[2]))
        
        for arm_i in range(actions.shape[0]):
            arm_state = int(o[arm_i])
            for arm_a in range(actions.shape[1]):
                param_upper = self.nature_params[arm_i, arm_state, arm_a, 1]
                actions[arm_i, arm_a] = param_upper

        return actions

    def bound_nature_actions(self, actions, state=None, reshape=True):
        return actions

    def get_policy_array(self, state_dim=0, N=0):

        N = self.nature_params.shape[0]
        S = self.nature_params.shape[1]
        A = self.nature_params.shape[2]

        all_states = list(itertools.product(np.arange(S), repeat=N))

        policy_array = np.zeros((len(all_states),N*A),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        for i, state in enumerate(all_states):
            policy_array[i] = self.get_nature_action(state).reshape(-1)

        return policy_array, tup_to_ind


class MiddleNaturePolicy:
    def __init__(self, nature_params, ind, perturbations=None, perturbation_size=0.1):
        self.nature_params=nature_params
        self.ind = ind
        self.perturbations = perturbations
        self.perturbation_size = perturbation_size
        self.name="Middle_Nature"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        actions = np.zeros((self.nature_params.shape[0],self.nature_params.shape[2]))
        
        for arm_i in range(actions.shape[0]):
            arm_state = int(o[arm_i])
            for arm_a in range(actions.shape[1]):
                param_mean = self.nature_params[arm_i, arm_state, arm_a].mean()
                
                if self.perturbations is not None:
                    param_range = np.ptp(self.nature_params[arm_i, arm_state, arm_a])
                    perturb_width = param_range*self.perturbation_size
                    perturbation = self.perturbations[arm_i, arm_state, arm_a]
                    perturbation = perturbation*perturb_width*2 - perturb_width
                    print(self.nature_params[arm_i, arm_state, arm_a])
                    print(perturbation)
                    print('before',param_mean)    
                    param_mean = param_mean + perturbation
                    print('after',param_mean)
                    print()

                actions[arm_i, arm_a] = param_mean



        return actions

    def bound_nature_actions(self, actions, state=None, reshape=True):
        return actions


    def get_policy_array(self, state_dim=0, N=0):

        N = self.nature_params.shape[0]
        S = self.nature_params.shape[1]
        A = self.nature_params.shape[2]

        all_states = list(itertools.product(np.arange(S), repeat=N))

        policy_array = np.zeros((len(all_states),N*A),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        for i, state in enumerate(all_states):
            policy_array[i] = self.get_nature_action(state).reshape(-1)

        return policy_array, tup_to_ind



class SampledRandomNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.param_setting=None
        self.ind = ind
        self.name="Sampled_Random_Nature"

    # only run this once
    def sample_param_setting(self, seed):
        assert self.param_setting is None
        
        rand_state = np.random.RandomState()
        rand_state.seed(seed)
        shape = self.nature_params.shape[:-1]
        sample = rand_state.rand(*shape)
        
        range_upper = self.nature_params[:, :, :, 1]
        range_lower = self.nature_params[:, :, :, 0]
        sample = sample*(range_upper - range_lower) + range_lower

        self.param_setting = sample


    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        actions = np.zeros((self.nature_params.shape[0],self.nature_params.shape[2]))
        
        for arm_i in range(actions.shape[0]):
            arm_state = int(o[arm_i])
            for arm_a in range(actions.shape[1]):
                actions[arm_i, arm_a] = self.param_setting[arm_i, arm_state, arm_a]

        return actions

    def bound_nature_actions(self, actions, state=None, reshape=True):
        return actions

    # we'll just settle for getting one sample from each...
    def get_policy_array(self, state_dim=0, N=0):

        N = self.nature_params.shape[0]
        S = self.nature_params.shape[1]
        A = self.nature_params.shape[2]

        all_states = list(itertools.product(np.arange(S), repeat=N))

        policy_array = np.zeros((len(all_states),N*A),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        for i, state in enumerate(all_states):
            policy_array[i] = self.get_nature_action(state).reshape(-1)

        return policy_array, tup_to_ind
