# Authors: Jackson A. Killian, 4 July, 2021
# 
# Adapted from repository by: OpenAI
#    - https://spinningup.openai.com/en/latest/

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# from numba import jit
from itertools import product
# from code.lp_methods import action_knapsack

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)



class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.


class MLPQCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q_net = mlp([obs_dim+act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, x):
        return torch.squeeze(self.q_net(x), -1) # Critical to ensure Q has right shape.


class MLPLambdaNet(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.lambda_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.lambda_net(obs), -1) 


# @jit(nopython=True)
def list_valid_action_combinations(N,C,B,options):

    costs = np.zeros(options.shape[0],dtype=np.float32)
    for i in range(options.shape[0]):
        costs[i] = C[options[i]].sum()
    valid_options = costs <= B
    options = options[valid_options]
    return options

class MLPActorCriticRMAB(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), C=None, N=None, B=None,
                 strat_ind=0, one_hot_encode=True, non_ohe_obs_dim=None,
                 state_norm=None,
                 activation=nn.Tanh):
        super().__init__()

        # one-hot-encode the states for now
        self.obs_dim = observation_space.shape[0]
        if not one_hot_encode:
            self.obs_dim = non_ohe_obs_dim

        self.act_type = 'd' # for discrete

        self.non_ohe_obs_dim = non_ohe_obs_dim
        self.one_hot_encode = one_hot_encode

        # we will only work with discrete actions
        self.act_dim = action_space.shape[0]

        self.pi_list = np.zeros(N,dtype=object)
        self.v_list = np.zeros(N,dtype=object)
        self.q_list = np.zeros(N,dtype=object)
        self.N = N
        self.C = C
        self.B = B
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.state_norm = state_norm

        for i in range(N):
            self.pi_list[i] = MLPCategoricalActor(self.obs_dim+1, self.act_dim, hidden_sizes, activation)
            self.v_list[i]  = MLPCritic(self.obs_dim+1, hidden_sizes, activation)
            self.q_list[i] = MLPQCritic(self.obs_dim+1, self.act_dim, hidden_sizes, activation)

        # Lambda_net is currently expected one input per arm, but other
        # networks are one-hot encoding the states...
        # This leads to a dimension difference...
        # need to change this eventually
        lambda_hidden_sizes = [8,8]
        self.lambda_net = MLPLambdaNet(N, lambda_hidden_sizes, activation)

        self.name = "RMABPPO"
        self.ind = strat_ind


    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)


    def reset_actor_and_critic_networks(self):
        for i in range(self.N):
            self.pi_list[i] = MLPCategoricalActor(self.obs_dim+1, self.act_dim, self.hidden_sizes, self.activation)
            self.v_list[i]  = MLPCritic(self.obs_dim+1, self.hidden_sizes, self.activation)
            self.q_list[i] = MLPQCritic(self.obs_dim+1, self.act_dim, self.hidden_sizes, self.activation)


    def return_large_lambda_loss(self, obs, gamma):

        disc_cost = 2 * self.B/(1-gamma)
        lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))

        loss = lamb*(self.B/(1-gamma) - disc_cost)

        return loss


    def step(self, obs, lamb):
        with torch.no_grad():
            if not self.one_hot_encode:
                obs = obs/self.state_norm

            a_list = np.zeros(self.N,dtype=int)
            v_list = np.zeros(self.N)
            logp_a_list = np.zeros(self.N)
            q_list = np.zeros(self.N)
            a1_probs = np.zeros(self.N)
            # a1_probs = np.zeros(self.N)

            for i in range(self.N):

                full_obs = None
                if self.one_hot_encode:
                    ohs = np.zeros(self.obs_dim)
                    ohs[int(obs[i])] = 1
                    full_obs = np.concatenate([ohs,[lamb]])
                else:
                    full_obs = np.concatenate([[obs[i]],[lamb]])

                full_obs = torch.as_tensor(full_obs,dtype=torch.float32)
                pi = self.pi_list[i]._distribution(full_obs)
                a = pi.sample()
                a1_probs[i] = pi.probs.numpy()[1]
                logp_a = self.pi_list[i]._log_prob_from_distribution(pi, a)
                v = self.v_list[i](full_obs)
                # print(logp_a)
                # print(obs, pi.probs, a, v)
                # print(pi.probs)
                # print(obs)
                # print(a)
                # print(pi.sample())
                # print(pi.sample())
                # print('above this!')
                # oha = np.zeros(self.act_dim)
                # oha[int(a)] = 1
                # x = torch.as_tensor(np.concatenate([full_obs, oha]), dtype=torch.float32)
                # q = self.q_list[i](x)

                a_list[i] = a.numpy()
                v_list[i] = v.numpy()
                logp_a_list[i] = logp_a.numpy()
                # q_list[i] = q.numpy()

        return a_list, v_list, logp_a_list, q_list, a1_probs

    def get_probs_for_all(self, obs, lamb):

        with torch.no_grad():
            prob_a_list = np.zeros(self.N)
            if not self.one_hot_encode:
                obs = obs/self.state_norm

            for i in range(self.N):
                full_obs = None
                if self.one_hot_encode:
                    ohs = np.zeros(self.obs_dim)
                    ohs[int(obs[i])] = 1
                    full_obs = np.concatenate([ohs,[lamb]])
                else:
                    full_obs = np.concatenate([[obs[i]],[lamb]])

                full_obs = torch.as_tensor(full_obs,dtype=torch.float32)
                pi = self.pi_list[i]._distribution(full_obs)
                prob_a_list[i] = pi.probs[1]
               

        return prob_a_list

        

    def act(self, obs, lamb):
        a = self.step(obs, lamb)[0]
        return a

    def act_test(self, obs):
        obs=obs.reshape(-1)
        # return self.act_test_deterministic(obs)
        return self.act_test_deterministic_multiaction(obs)
        # return self.act_test_stochastic_multi_action(obs)

    def get_lambda(self, obs):
        obs = obs.reshape(-1)
        if not self.one_hot_encode:
            obs = obs/self.state_norm
        lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))
        return lamb.detach().numpy()

    # Currently only implemented for binary action
    def act_test_deterministic(self, obs):
        # print("Enforcing budget constraint on action")
        ACTION = 1
        a_list = np.zeros(self.N,dtype=int)
        pi_list = np.zeros((self.N,self.act_dim),dtype=float)
        with torch.no_grad():    
            if not self.one_hot_encode:
                obs = obs/self.state_norm

            lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))

            for i in range(self.N):
                full_obs = None
                if self.one_hot_encode:
                    ohs = np.zeros(self.obs_dim)
                    ohs[int(obs[i])] = 1
                    full_obs = np.concatenate([ohs,[lamb]])
                else:
                    full_obs = np.concatenate([[obs[i]],[lamb]])

                full_obs = torch.as_tensor(full_obs, dtype=torch.float32)

                pi = self.pi_list[i]._distribution(full_obs).probs.detach().numpy()
                pi_list[i] = pi

            # play the actions with the largest probs
            a1_list = pi_list[:,ACTION]
            # print(a1_list)

            sorted_inds = np.argsort(a1_list)[::-1]

            i = 0
            budget_spent = 0
            while budget_spent < self.B and i < self.N:

                # if taking the next action (greedy) puts over budget, break
                if budget_spent + self.C[ACTION] > self.B:
                    break

                a_list[sorted_inds[i]] = ACTION
                budget_spent += self.C[ACTION]

                i+=1

                
        return a_list

    # Multi-action implementation
    # Naive -- do the same thing, but first take a max over the actions per arm
    def act_test_deterministic_multiaction(self, obs):
        # print("Enforcing budget constraint on action")

        actions = np.zeros(self.N,dtype=int)
        pi_list = np.zeros((self.N,self.act_dim),dtype=float)
        with torch.no_grad(): 
            # print(obs)   
            if not self.one_hot_encode:
                obs = obs/self.state_norm

            lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))

            for i in range(self.N):
                full_obs = None
                if self.one_hot_encode:
                    ohs = np.zeros(self.obs_dim)
                    ohs[int(obs[i])] = 1
                    full_obs = np.concatenate([ohs,[lamb]])
                else:
                    full_obs = np.concatenate([[obs[i]],[lamb]])
                # print(full_obs)
                full_obs = torch.as_tensor(full_obs, dtype=torch.float32)

                pi = self.pi_list[i]._distribution(full_obs).probs.detach().numpy()
                pi_list[i] = pi
                # print(pi)

            # sort each list by its most probable action
            # then play them in order
            
            # a1_list = pi_list[:,ACTION]
            # print(a1_list)

            row_maxes = pi_list.max(axis=1)
            row_order = np.argsort(row_maxes)[::-1]

            pi_arg_maxes = np.argsort(pi_list, axis=1)

            actions = np.zeros(self.N, dtype=int)

            budget_spent = 0
            done = False

            while budget_spent < self.B and not done:

                i=0
                while i < self.N:
                    arm = row_order[i]
                    arm_a = pi_arg_maxes[row_order[i]][-1]

                    a_cost = self.C[arm_a]
                    
                    # if difference in price takes us over, we have to stop
                    # print(actions)
                    # print(arm)
                    # print(actions[arm])
                    # print(C)
                    # print('budget',budget_spent, 'cost', a_cost, 'action', arm_a, 'arm',arm, 'c',C[actions[arm]])
                    if budget_spent + a_cost - self.C[actions[arm]] > self.B:
                        done = True
                        # print('broke here')
                        break
                    else:
                        i+=1
                        # assign action
                        actions[arm] = arm_a
                        # now hide all cheaper actions
                        pi_list[arm, :arm_a+1] = 0
                        # print(a)

                        budget_spent = sum(self.C[a] for a in actions)

                        # also hide all actions that are now too expensive
                        cost_diff_array = np.zeros(pi_list.shape)
                        # print('actions',actions)
                        for j in range(self.N):
                            cost_diff_array[j] = self.C - self.C[actions[j]]
                        # print('a\n',a_og)
                        # print('nowa\n',a)
                        # print('costdif\n',cost_diff_array)
                        # print('b',B)
                        # print('budget',budget_spent)
                        overbudget_action_inds = cost_diff_array > self.B - budget_spent
                        # print('inds')
                        # print(overbudget_action_inds)
                        
                        if overbudget_action_inds.any():
                            i = 0
                            pi_list[overbudget_action_inds] = 0
                            row_maxes = pi_list.max(axis=1)
                            row_order = np.argsort(row_maxes)[::-1]

                            pi_arg_maxes = np.argsort(pi_list, axis=1)
                            # print('maxes',a_maxes)
                            # print('order',row_order)
                            # print('maxes',a_arg_maxes)
                        if not pi_list.sum() > 0:
                            done = True
                            break


                row_maxes = pi_list.max(axis=1)
                row_order = np.argsort(row_maxes)[::-1]

                pi_arg_maxes = np.argsort(pi_list, axis=1)

        # print(actions)                
        return actions

    # Multi-action implementation
    # Not implemented yet...
    def act_test_deterministic_knapsack(self, obs):
        # print("Enforcing budget constraint on action")

        a_list = np.zeros(self.N,dtype=int)
        pi_list = np.zeros((self.N,self.act_dim),dtype=float)
        with torch.no_grad():    
            if not self.one_hot_encode:
                obs = obs/self.state_norm

            lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))

            for i in range(self.N):
                full_obs = None
                if self.one_hot_encode:
                    ohs = np.zeros(self.obs_dim)
                    ohs[int(obs[i])] = 1
                    full_obs = np.concatenate([ohs,[lamb]])
                else:
                    full_obs = np.concatenate([[obs[i]],[lamb]])
                full_obs = torch.as_tensor(full_obs, dtype=torch.float32)

                pi = self.pi_list[i]._distribution(full_obs).probs.detach().numpy()
                pi_list[i] = pi

            

                
        return a_list

    # Currently only implemented for binary action
    def act_test_stochastic_binary(self, obs):
        # print("Enforcing budget constraint on action")
        ACTION = 1
        a_list = np.zeros(self.N,dtype=int)
        pi_list = np.zeros((self.N,self.act_dim),dtype=float)
        obs = obs.reshape(-1)

        with torch.no_grad():   
            if not self.one_hot_encode:
                    obs = obs/self.state_norm 

            lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))

            for i in range(self.N):
                full_obs = None
                if self.one_hot_encode:
                    ohs = np.zeros(self.obs_dim)
                    ohs[int(obs[i])] = 1
                    full_obs = np.concatenate([ohs,[lamb]])
                else:
                    full_obs = np.concatenate([[obs[i]],[lamb]])
                full_obs = torch.as_tensor(full_obs, dtype=torch.float32)

                pi = self.pi_list[i]._distribution(full_obs).probs.detach().numpy()
                pi_list[i] = pi

            # play the actions with the largest probs
            a1_list = pi_list[:,ACTION]

            options = list(np.arange(self.N))

            i = 0
            budget_spent = 0
            while budget_spent < self.B and i < self.N:

                # if taking the next action (greedy) puts over budget, break
                if budget_spent + self.C[ACTION] > self.B:
                    break

                normalized_probs = a1_list[options] / a1_list[options].sum()

                # might want to give the ac object its own seed? 
                # Rather than relying on the numpy global seed
                # print(options, normalized_probs)
                choice = np.random.choice(options, p=normalized_probs)

                a_list[choice] = ACTION
                budget_spent += self.C[ACTION]

                options.remove(choice)

                i+=1

                
        return a_list


    # multi-action implementation
    def act_test_stochastic_multi_action(self, obs):
        # print("Enforcing budget constraint on action")

        actions = np.zeros(self.N,dtype=int)
        pi_list = np.zeros((self.N,self.act_dim),dtype=float)
        obs = obs.reshape(-1)

        with torch.no_grad():   
            if not self.one_hot_encode:
                    obs = obs/self.state_norm 

            lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))

            for i in range(self.N):
                full_obs = None
                if self.one_hot_encode:
                    ohs = np.zeros(self.obs_dim)
                    ohs[int(obs[i])] = 1
                    full_obs = np.concatenate([ohs,[lamb]])
                else:
                    full_obs = np.concatenate([[obs[i]],[lamb]])
                full_obs = torch.as_tensor(full_obs, dtype=torch.float32)

                pi = self.pi_list[i]._distribution(full_obs).probs.detach().numpy()
                pi_list[i] = pi

            # play the actions with the largest probs
            

            # options = list(np.arange(self.N))

            # actions = np.zeros(N,dtype=int)

            current_action_cost = 0
            p=pi_list.max(axis=1)
            p = p/p.sum()
            process_order = np.random.choice(np.arange(self.N), self.N, p=p, replace=False)
            for arm in process_order:
                
                # select an action at random
                num_valid_actions_left = len(self.C[self.C<=self.B-current_action_cost])
                p = pi_list[arm][self.C<=self.B-current_action_cost]
                p = p/p.sum()
                p = None
                a = np.random.choice(np.arange(num_valid_actions_left), p=p)
                current_action_cost += self.C[a]
                # if the next selection takes us over budget, break
                if current_action_cost > self.B:
                    break
                # print(actions)

                actions[arm] = a


        return actions


    def act_q(self, obs):
        with torch.no_grad():
            max_q = -10e10
            action = 0
            for ind,row in enumerate(np.eye(self.act_dim)):
                x = torch.as_tensor(np.concatenate([obs, row]), dtype=torch.float32)
                q = self.q(x)
                if q >= max_q:
                    max_q = q
                    action = ind
        #         print(ind, q)
        #     print()
        # print(action)
        return action


