import numpy as np
import itertools


# Don't use if you need deterministic (e.g., in main loop of double oracle)
class RandomNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.ind = ind
        self.name="Random_Nature_c"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        actions = np.zeros(self.nature_params.shape[0])
        
        for arm_i in range(actions.shape[0]):
            param_range = self.nature_params[arm_i, 1] - self.nature_params[arm_i, 0]
            param_lower = self.nature_params[arm_i, 0]
            actions[arm_i] = np.random.rand()*param_range + param_lower

        return actions

    def bound_nature_actions(self, actions, state=None, reshape=True):
        return actions

    # we'll just settle for getting one sample from each...
    def get_policy_array(self, state_dim=0, N=0):

        N = self.nature_params.shape[0]
        S = state_dim

        all_states = list(itertools.product(np.arange(S), repeat=N))

        policy_array = np.zeros((len(all_states),N),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        for i, state in enumerate(all_states):
            policy_array[i] = self.get_nature_action(state)

        return policy_array, tup_to_ind


class PessimisticNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.ind = ind
        self.name="Pessimist_Nature_c"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):

        return self.nature_params.min(axis=1)

    def bound_nature_actions(self, actions, state=None, reshape=True):
        return actions

    def get_policy_array(self, state_dim=0, N=0):

        N = self.nature_params.shape[0]
        S = state_dim

        all_states = list(itertools.product(np.arange(S), repeat=N))

        policy_array = np.zeros((len(all_states),N),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        policy_array[:] = self.get_nature_action(None)

        return policy_array, tup_to_ind

class OptimisticNaturePolicy:
    def __init__(self, nature_params, ind):
        self.nature_params=nature_params
        self.ind = ind
        self.name="Optimist_Nature_c"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        return self.nature_params.max(axis=1)

    def bound_nature_actions(self, actions, state=None, reshape=True):
        return actions

    def get_policy_array(self, state_dim=0, N=0):

        N = self.nature_params.shape[0]
        S = state_dim

        all_states = list(itertools.product(np.arange(S), repeat=N))

        policy_array = np.zeros((len(all_states),N),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        policy_array[:] = self.get_nature_action(None)

        return policy_array, tup_to_ind




class MiddleNaturePolicy:
    def __init__(self, nature_params, ind, perturbations=None, perturbation_size=0.1):
        self.nature_params=nature_params
        self.ind = ind
        self.perturbations = perturbations
        self.perturbation_size = perturbation_size
        self.name="Middle_Nature_c"

    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        
        a = self.nature_params.mean(axis=1)
        if self.perturbations is not None:
            param_range = np.ptp(self.nature_params, axis=1)
            perturb_width = param_range*self.perturbation_size
            perturbation = self.perturbations
            perturbation = perturbation*perturb_width*2 - perturb_width

        return a


    def bound_nature_actions(self, actions, state=None, reshape=True):
        return actions


    def get_policy_array(self, state_dim=0, N=0):

        N = self.nature_params.shape[0]
        S = state_dim

        all_states = list(itertools.product(np.arange(S), repeat=N))

        policy_array = np.zeros((len(all_states),N),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        policy_array[:] = self.get_nature_action(None)

        return policy_array, tup_to_ind



class DetermNaturePolicy:
    def __init__(self, param_setting, tok):
        self.param_setting=param_setting
        self.tok = tok
        self.name="Determ_Nature"

    def __repr__(self):
        return "%s_%s"%(self.name, self.tok)

    def get_nature_action(self, o):
        
        return self.param_setting

    def bound_nature_actions(self, actions, state=None, reshape=True):
        return actions

    # A - 10 in A - 0, middle
    # rangeA = [0, 1]

    # B - 10 in B - 1, bottom
    # rangeB = [0.05, 0.9]
    # rangeB = [0.05, 0.8] # nudge the middle a bit lower so RL learns the middle policy exactly

    # C - 30 in C - 2, top
    # rangeC = [0.1, 0.95]
    # rangeC = [0.2, 0.95] # nudge the middle a bit higher so RL learns the middle policy exactly

    # pess hawkins policy order: [2,1,0]
    # so maximize returns from a policy that goes [0,1,2]
    # for now, just max the prob of the first entry, min probs of remaining entries
    def pess_regret_oracle(self):
        self.param_setting = np.array([1, 0.05, 0.1])
        # self.param_setting = np.array([1, 0.05, 0.2])

    # mid hawkins policy order: [2,0,1]
    # so maximize returns from a policy that goes [1,0,2]
    def mid_regret_oracle(self):
        self.param_setting = np.array([0, 0.9, 0.1])
        # self.param_setting = np.array([0, 0.8, 0.2])

    # mid hawkins policy order: [0,2,1]
    # so maximize returns from a policy that goes [1,2,0]
    def opt_regret_oracle(self):
        self.param_setting = np.array([0, 0.9, 0.1])
        # self.param_setting = np.array([0, 0.8, 0.2])


    def get_policy_array(self, state_dim=0, N=0):

        N = self.param_setting.shape[0]
        S = state_dim

        all_states = list(itertools.product(np.arange(S), repeat=N))

        policy_array = np.zeros((len(all_states),N),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = np.array(all_states)

        policy_array[:] = self.get_nature_action(None)

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
        
        range_upper = self.nature_params[:, 1]
        range_lower = self.nature_params[:, 0]
        sample = sample*(range_upper - range_lower) + range_lower

        self.param_setting = sample


    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)

    def get_nature_action(self, o):
        return self.param_setting

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

