# Authors: Jackson A. Killian, 4 July, 2021
# 
# Adapted from repository by: OpenAI
#    - https://spinningup.openai.com/en/latest/

import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# from numba import jit
from itertools import product


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




class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        ### TODO - check what this variance looks like
        # print(self.log_std)
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


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

class RMABLambdaNatureOracle(nn.Module):


    def __init__(self, observation_space, action_space_agent, nature_parameter_ranges,
                    action_dim_nature, env,
                 hidden_sizes=(64,64), C=None, N=None, B=None, one_hot_encode=True, non_ohe_obs_dim=None,
                 state_norm=1, nature_state_norm=1,
                 strat_ind=0,
                 activation=nn.Tanh):
        super().__init__()

        # one-hot-encode the states for now
        self.observation_space = observation_space
        self.action_space_agent = action_space_agent
        self.nature_parameter_ranges = nature_parameter_ranges
        self.obs_dim = observation_space.shape[0]
        self.act_type = 'd' # for discrete
        if not one_hot_encode:
            self.obs_dim = non_ohe_obs_dim

        self.non_ohe_obs_dim = non_ohe_obs_dim
        self.one_hot_encode = one_hot_encode

        # we will only work with discrete actions
        self.act_dim_agent = action_space_agent.shape[0]
        self.act_dim_nature = action_dim_nature

        self.pi_list_agent = np.zeros(N,dtype=object)
        # self.pi_list_nature = np.zeros(N,dtype=object)
        self.v_list_agent = np.zeros(N,dtype=object)
        # self.v_list_nature = np.zeros(N,dtype=object)
        self.q_list_agent = np.zeros(N,dtype=object)
        # self.q_list_nature = np.zeros(N,dtype=object)
        self.N = N
        self.C = C
        self.B = B
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.state_norm = state_norm
        self.nature_state_norm = nature_state_norm

        self.env = env

        # right now only accepts 1-D state and action inputs
        # input is one state per arm
        # output is the aciton dimension, usually a factor of number of arms and actions
        self.pi_nature = MLPGaussianActor(self.N, self.act_dim_nature, hidden_sizes, activation)
        # takes the action space of the actors...
        self.v_nature = MLPCritic(self.N + self.N, hidden_sizes, activation)
        self.q_nature  = MLPQCritic(self.N + self.N, self.act_dim_nature, hidden_sizes, activation)

        for i in range(N):
            # +1 because we add lambda as an input
            self.pi_list_agent[i] = MLPCategoricalActor(self.obs_dim+1, self.act_dim_agent, hidden_sizes, activation)
            self.v_list_agent[i]  = MLPCritic(self.obs_dim + self.act_dim_nature + 1, hidden_sizes, activation)
            self.q_list_agent[i]  = MLPQCritic(self.obs_dim + self.act_dim_nature + 1, self.act_dim_agent, hidden_sizes, activation)
            

        # Lambda_net is currently expected one input per arm, but other
        # networks are one-hot encoding the states...
        # This leads to a dimension difference...
        # need to change this eventually
        lambda_hidden_sizes = [8,8]
        self.lambda_net = MLPLambdaNet(N, lambda_hidden_sizes, activation)



        self.name = "MA-RMABPPO"
        self.ind = strat_ind


    def __repr__(self):
        return "%s_%i"%(self.name, self.ind)


    def reset_actor_and_critic_networks(self):
        # right now only accepts 1-D state and action inputs
        self.pi_nature = MLPGaussianActor(self.N, self.act_dim_nature, hidden_sizes, activation)
        # takes the action space of the actors...
        self.v_nature = MLPCritic(self.N + self.N, hidden_sizes, activation)
        self.q_nature  = MLPQCritic(self.N + self.N, self.act_dim_nature, hidden_sizes, activation)

        for i in range(N):
            # +1 because we add lambda as an input
            self.pi_list_agent[i] = MLPCategoricalActor(self.obs_dim+1, self.act_dim_agent, hidden_sizes, activation)
            self.v_list_agent[i]  = MLPCritic(self.obs_dim + self.act_dim_nature + 1, hidden_sizes, activation)
            self.q_list_agent[i]  = MLPQCritic(self.obs_dim + self.act_dim_nature + 1, self.act_dim_agent, hidden_sizes, activation)
            

    def return_large_lambda_loss(self, obs, gamma):

        disc_cost = 2 * self.B/(1-gamma)
        lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))

        loss = lamb*(self.B/(1-gamma) - disc_cost)

        return loss

    # this is easier to attach to environment code
    def bound_nature_actions(self, a_nature, state=None, reshape=True):
        
        return self.env.bound_nature_actions(a_nature, state=state, reshape=reshape)

    def get_nature_action(self,obs):
        
        # return self.get_nature_action_stochastic(obs)
        return self.get_nature_action_deterministic(obs)


    def get_nature_action_stochastic(self,obs):
         with torch.no_grad():

            pi_nature = self.pi_nature._distribution(obs)
            a_nature = pi_nature.sample()
            return a_nature


    def get_nature_action_deterministic(self,obs):
         with torch.no_grad():
            if not self.one_hot_encode:
                obs = obs/self.state_norm/self.nature_state_norm
            mu = self.pi_nature.mu_net(obs)
            return mu.numpy()


    def step(self, obs, lamb):
        with torch.no_grad():
            if not self.one_hot_encode:
                obs = obs/self.state_norm

            a_agent_list = np.zeros(self.N,dtype=int)
            logp_a_agent_list = np.zeros(self.N)

            v_agent_list = np.zeros(self.N)
            q_agent_list = np.zeros(self.N)

            a1_agent_probs = np.zeros(self.N)


            # nature action
            pi_nature = self.pi_nature._distribution(obs/self.nature_state_norm)
            a_nature = pi_nature.sample()
            logp_a_nature = self.pi_nature._log_prob_from_distribution(pi_nature, a_nature)

            
            # agent action and value function
            for i in range(self.N):
                full_obs = None
                if self.one_hot_encode:
                    ohs = np.zeros(self.obs_dim)
                    ohs[int(obs[i])] = 1
                    full_obs = np.concatenate([ohs,[lamb]])
                else:
                    full_obs = np.concatenate([[obs[i]],[lamb]])

                full_obs = torch.as_tensor(full_obs,dtype=torch.float32)
                pi_agent = self.pi_list_agent[i]._distribution(full_obs)
                a1_agent_probs[i] = pi_agent.probs.numpy()[1]
                a = pi_agent.sample()
                logp_a = self.pi_list_agent[i]._log_prob_from_distribution(pi_agent, a)
                a_agent_list[i] = a.numpy()
                logp_a_agent_list[i] = logp_a.numpy()

                a_nature_bounded = self.env.bound_nature_actions(a_nature, state=obs.numpy(), reshape=False)
                x_s_a_nature = torch.as_tensor(np.concatenate([full_obs, a_nature_bounded]), dtype=torch.float32)

                # v_agent takes nature's actions as input
                v_agent_list[i] = self.v_list_agent[i](x_s_a_nature)

                # x_all = torch.as_tensor(np.concatenate([full_obs, a_nature_bounded, [a]]), dtype=torch.float32)
                q_agent_list[i] = 0#self.q_list_agent[i](x_all)

            # nature value function
            x_s_a_agent = torch.as_tensor(np.concatenate([obs, a_agent_list]), dtype=torch.float32)
            v_nature = self.v_nature(x_s_a_agent)

            x_all = torch.as_tensor(np.concatenate([obs, a_agent_list, a_nature]), dtype=torch.float32)
            q_nature = 0#self.v_nature(x_all)
            
                

        # return a_agent_list, v_agent_list, logp_a_agent_list, q_agent_list, a_nature.numpy(), v_nature.numpy(), logp_a_nature.numpy(), q_nature#.numpy(), 
        return a_agent_list, v_agent_list, logp_a_agent_list, q_agent_list, a_nature.numpy(), v_nature.numpy(), logp_a_nature.numpy(), q_nature, a1_agent_probs

    def get_probs_for_all(self, obs, lamb):

        with torch.no_grad():
            if not self.one_hot_encode:
                obs = obs/self.state_norm
            prob_a_list = np.zeros(self.N)

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
        return self.act_test_deterministic(obs)

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

            lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))

            for i in range(self.N):
                ohs = np.zeros(self.obs_dim)
                ohs[int(obs[i])] = 1
                full_obs = np.concatenate([ohs,[lamb]])
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

    # Currently only implemented for binary action
    def act_test_stochastic(self, obs):
        # print("Enforcing budget constraint on action")
        ACTION = 1
        a_list = np.zeros(self.N,dtype=int)
        pi_list = np.zeros((self.N,self.act_dim),dtype=float)
        obs = obs.reshape(-1)

        with torch.no_grad():    

            lamb = self.lambda_net(torch.as_tensor(obs,dtype=torch.float32))

            for i in range(self.N):
                ohs = np.zeros(self.obs_dim)
                ohs[int(obs[i])] = 1
                full_obs = np.concatenate([ohs,[lamb]])
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
                # print(ind, q)
            # print()
        # print(action)
        return action


    def get_policy_array(self, state_dim=0, N=0):
        

        all_states = list(product(np.arange(self.observation_space.shape[0]), repeat=self.N))

        policy_array = np.zeros((len(all_states),self.act_dim_nature),dtype=float)

        tup_to_ind = dict(zip(all_states,np.arange(len(all_states))))

        all_states = torch.as_tensor(all_states, dtype=torch.float32)

        for i in range(all_states.shape[0]):
            actions = self.get_nature_action(all_states[i])
            actions = self.env.bound_nature_actions(actions, state=all_states[i].numpy(), reshape=False)
            policy_array[i] = actions

        return policy_array, tup_to_ind


