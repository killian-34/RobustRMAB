# Authors: Jackson A. Killian, Lily Xu. 3 July, 2021
# 
# Adapted from repository by: Lily Xu
#    - From UAI'21 paper: Xu, Lily, et al. "Robust Reinforcement Learning Under Minimax Regret for 
#      Green Security." arXiv preprint arXiv:2106.08413 (2021).



import sys, os
import time
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt

from robust_rmab.algos.rmabppo.agent_oracle import AgentOracle
from robust_rmab.algos.ma_rmabppo.nature_oracle import NatureOracle

from robust_rmab.nfg_solver import solve_minimax_regret, get_payoff, solve_minimax_regret_with_regret_array

from robust_rmab.environments.bandit_env_robust import CounterExampleRobustEnv, ARMMANRobustEnv, SISRobustEnv

from robust_rmab.baselines.agent_baselines import   (
                            PessimisticAgentPolicy, RandomAgentPolicy
                        )

from robust_rmab.baselines.nature_baselines_armman import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy
                        )




def compute_mean_l2_diff(nsp1, nsp2):
    n = len(nsp1)
    sum_diff = 0
    for i in range(len(nsp1)):
        # print(nsp1[i], nsp2[i])
        # print(np.linalg.norm(nsp1[i]-nsp2[i]))
        # print()
        sum_diff += np.linalg.norm(nsp1[i]-nsp2[i])
    mean_diff = sum_diff/n
    # print(sum_diff, mean_diff)
    return mean_diff


class DoubleOracle:
    def __init__(self, data, N, budget, horizon, max_epochs_double_oracle,
                    S=0, A=0, seed=0, reward_bound=1, n_cpu=1, n_simu_epochs=50,
                    home_dir="", exp_name="", gamma = 0.99,
                    pop_size=0, one_hot_encode=True, state_norm=1,
                    non_ohe_obs_dim=0,
                    agent_kwargs=dict(), nature_kwargs=dict()):

        self.data        = data
        self.N           = N
        self.budget      = budget
        self.horizon     = horizon
        self.gamma       = gamma
        # self.n_agent_epochs = n_agent_epochs
        # self.n_nature_epochs = n_nature_epochs
        self.max_epochs_double_oracle = max_epochs_double_oracle
        self.n_simu_epochs = n_simu_epochs

        self.S=S
        self.A=A
        self.seed=seed
        self.reward_bound=reward_bound
        self.home_dir=home_dir
        self.exp_name=exp_name
        self.n_cpu = n_cpu

        self.pop_size = pop_size
        self.one_hot_encode = one_hot_encode
        self.non_ohe_obs_dim = non_ohe_obs_dim
        self.state_norm = state_norm

        self.nature_state_norm = 1

        if data == 'random':
            self.env_fn = lambda : RandomBanditEnv(N,S,A,budget,seed,reward_bound)

        if data == 'random_reset':
            self.env_fn = lambda : RandomBanditResetEnv(N,S,A,budget,seed,reward_bound)

        if data == 'armman':
            self.env_fn = lambda : ARMMANRobustEnv(N,budget,seed)

        if data == 'circulant':
            self.env_fn = lambda : CirculantDynamicsEnv(N,budget,seed)

        if data == 'counterexample':
            self.env_fn = lambda : CounterExampleRobustEnv(N,budget,seed)

        if data == 'sis':
            self.env_fn = lambda : SISRobustEnv(N,budget,pop_size,seed)
            self.nature_state_norm = 1


        self.env = self.env_fn()
        self.sampled_nature_parameter_ranges = self.env.sample_parameter_ranges()
        # important to make sure these are always the same for all instatiations of the env
        self.env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges


        self.agent_oracle  = AgentOracle(data, N, S, A, budget, seed, reward_bound,
                             agent_kwargs=agent_kwargs, home_dir=home_dir, exp_name=exp_name,
                             sampled_nature_parameter_ranges = self.sampled_nature_parameter_ranges,
                             pop_size=self.pop_size, one_hot_encode=one_hot_encode, state_norm=state_norm,
                             non_ohe_obs_dim=non_ohe_obs_dim)
 

        self.nature_oracle = NatureOracle(data, N, S, A, budget, seed, reward_bound,
                             nature_kwargs=nature_kwargs, home_dir=home_dir, exp_name=exp_name,
                             sampled_nature_parameter_ranges = self.sampled_nature_parameter_ranges,
                             pop_size=self.pop_size, one_hot_encode=one_hot_encode, state_norm=state_norm,
                             non_ohe_obs_dim=non_ohe_obs_dim, nature_state_norm=self.nature_state_norm)


        

        strat_type_ind = 0
        pess_nature_pol = PessimisticNaturePolicy(self.sampled_nature_parameter_ranges, strat_type_ind)


        # initialize strategy sets
        self.agent_strategies  = []  # agent policy
        self.nature_strategies = [pess_nature_pol]  # nature policy
        self.payoffs           = [] # agent regret for each (agent strategy, nature strategy) combo

        strat_type_ind = 0
        pess_agent_pol = PessimisticAgentPolicy(self.N, strat_type_ind)

        # init the payoff matrix
        add_to_seed = 0
        self.update_payoffs_agent(pess_agent_pol, add_to_seed)

        


    def run(self):
        agent_eq    = np.ones(len(self.agent_strategies))/len(self.agent_strategies) # account for baselines
        # agent_eq    = np.array([1.]) # account for baselines
        nature_eq   = np.ones(len(self.nature_strategies))/len(self.nature_strategies)

        # repeat until convergence
        converged = False
        n_epochs = 0
        while not converged:
            print('-----------------------------------')
            print('epoch {}'.format(n_epochs))
            print('-----------------------------------')
            print('n_agent_strategies', len(self.agent_strategies))
            print('n_nature_strategies', len(self.nature_strategies))
            print('-----------------------------------')

            add_to_seed = 0#self.n_cpu*n_epochs

            # if first epoch, defender response is ideal defender for initial attractiveness
            agent_br = self.agent_oracle.best_response(self.nature_strategies, nature_eq, add_to_seed)
            # self.update_payoffs_agent(agent_br)
            nature_br = self.nature_oracle.best_response(self.agent_strategies, agent_eq, add_to_seed)
            # self.update_payoffs_nature(nature_br)


            add_to_seed = 0 # keep this zero so everyone gets the same seeds for the n_simu_epochs
            self.update_payoffs(nature_br, agent_br, add_to_seed)


            # find equilibrium of subgame
            agent_eq, nature_eq = self.find_equilibrium()

            print('agent equilibrium    ', np.round(agent_eq, 3))
            print('nature equilibrium ', np.round(nature_eq, 3))

            max_regret_game = np.array(self.payoffs) - np.array(self.payoffs).max(axis=0)
            print('!!!!! ', n_epochs, 'payoffs are', get_payoff(max_regret_game, agent_eq, nature_eq))

            if n_epochs >= self.max_epochs_double_oracle: 
                converged = True
                break

            n_epochs += 1

            assert len(self.payoffs) == len(self.agent_strategies), '{} payoffs, {} agent strategies'.format(len(self.payoffs), len(self.agent_strategies))
            assert len(self.payoffs[0]) == len(self.nature_strategies), '{} payoffs[0], {} nature strategies'.format(len(self.payoff[0]), len(self.nature_strategies))

        

        return agent_eq, nature_eq


    def compute_regret(self, agent_s, nature_s, max_reward):
        reward = self.agent_oracle.simulate_reward(agent_s, nature_s, display=False)
        regret = max_reward - reward
        # assert regret >= 0
        if regret < 0:
            print('  uh oh! regret is negative. max reward {:.3f}, reward {:.3f}'.format(max_reward, reward))
        return regret

    def compute_payoff_regret(self, agent_eq):
        """ given a agent mixed strategy, compute the expected regret in the payoff matrix """
        assert abs(sum(agent_eq) - 1) <= 1e-3

        regret = np.array(do.payoffs) - np.array(do.payoffs).max(axis=0)
        # if agent playing a pure strategy
        if len(np.where(agent_eq > 0)[0]) == 1:
            agent_strategy_i = np.where(agent_eq > 0)[0].item()
            strategy_regrets = regret[agent_strategy_i]
            return -np.min(strategy_regrets) # return max regret (min reward)
        else:
            raise Exception('need to implement')

    def find_equilibrium(self, ignore_rows = 0, ignore_cols = 0):
        
        payoffs = self.payoffs
        if ignore_rows > 0:
            payoffs = payoffs[:-ignore_rows]
        if ignore_cols > 0:
            payoffs = payoffs[:, :-ignore_rows]
        """ solve for minimax regret-optimal mixed strategy """
        agent_eq, nature_eq = solve_minimax_regret(payoffs)
        return agent_eq, nature_eq

    def find_equilibrium_with_regret_array(self, regret_array, ignore_rows = 0, ignore_cols = 0):

        payoffs = np.copy(regret_array)
        if ignore_rows > 0:
            payoffs = payoffs[:-ignore_rows]
        if ignore_cols > 0:
            payoffs = payoffs[:, :-ignore_rows]
        """ solve for minimax regret-optimal mixed strategy """
        agent_eq, nature_eq = solve_minimax_regret_with_regret_array(payoffs)
        return agent_eq, nature_eq

        

    def update_payoffs(self, nature_br, agent_br, add_to_seed):
        """ update payoff matrix (in place) """
        self.update_payoffs_agent(agent_br, add_to_seed)
        self.update_payoffs_nature(nature_br, add_to_seed)

    def update_payoffs_agent(self, agent_br, add_to_seed=0):
        """ update payoff matrix (only adding agent strategy)

        returns index of new strategy """
        self.agent_strategies.append(agent_br)

        # for new defender strategy: compute regret w.r.t. all nature strategies
        new_payoffs = []
        for i, nature_s in enumerate(self.nature_strategies):
            print('Simulating rewards, %s vs. %s' %(agent_br,nature_s) )
            reward = self.agent_oracle.simulate_reward(agent_br, nature_s, seed=self.seed+add_to_seed, 
                steps_per_epoch=self.horizon, epochs=self.n_simu_epochs, gamma=self.gamma)
            new_payoffs.append(reward)
        self.payoffs.append(new_payoffs)

        return len(self.agent_strategies) - 1

    def update_payoffs_nature(self, nature_br, add_to_seed=0):
        """ update payoff matrix (only adding nature strategy)

        returns index of new strategy """
        self.nature_strategies.append(nature_br)

        # update payoffs
        # for new nature strategy: compute regret w.r.t. all defender strategies
        for i, agent_s in enumerate(self.agent_strategies):
            print('Simulating rewards, %s vs. %s' %(agent_s,nature_br) )
            reward = self.agent_oracle.simulate_reward(agent_s, nature_br, seed=self.seed+add_to_seed, 
                steps_per_epoch=self.horizon, epochs=self.n_simu_epochs, gamma=self.gamma)
            self.payoffs[i].append(reward)

        return len(self.nature_strategies) - 1



###################################################
# baselines
###################################################

def use_middle(do, add_to_seed, ind, perturbations=None, perturbation_size=0.1):
    """ solve optimal reward relative to midpoint of uncertainty interval
    sequential policy, but based on the center of the uncertainty set """

    middle_nature_policy = MiddleNaturePolicy(do.sampled_nature_parameter_ranges, ind, 
                            perturbations=perturbations, perturbation_size=perturbation_size)
    agent_br = do.agent_oracle.best_response([middle_nature_policy], [1.], add_to_seed)
    return agent_br


def use_random(do, add_to_seed, ind):
    """ solve optimal reward relative to midpoint of uncertainty interval
    sequential policy, but based on the center of the uncertainty set """

    random_nature_policy = RandomNaturePolicy(do.sampled_nature_parameter_ranges, ind)
    agent_br = do.agent_oracle.best_response([random_nature_policy], [1.], add_to_seed)
    return agent_br



class RandomPolicy:
    def __init__(self, park_params):
        self.n_targets = park_params['n_targets']
        self.budget    = park_params['budget']

    def select_action(self, state):
        action = np.random.rand(self.n_targets)
        action /= action.sum() * self.budget
        return action


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Robust RMAB main code -- runs Double Oracle to compute Minimax regret")
    parser.add_argument('--hid', type=int, default=64, help="Number of units in each layer of the neural networks used for the Oracles")
    parser.add_argument('-l', type=int, default=2, help="Depth of the neural networks used for Agent and Nature Oracles (i.e., layers)")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--seed', '-s', type=int, default=0, help="Seed")
    parser.add_argument('--cpu', type=int, default=1, help="Must be set to 1 for now. Multi-processing is a TODO.")
    parser.add_argument('--max_epochs_double_oracle', type=int, default=5, help="Number of iterations to run the double oracle")
    parser.add_argument('--n_simu_epochs', type=int, default=50, help="Number of simulations to run for estimating each entry of the regret matrix (i.e., simulation of one Agent Pure strategy against one Nature Pure Strategy")
    parser.add_argument('--horizon', type=int, default=10, help="Horizon length of the environment (for estimating the regret matrix)")


    parser.add_argument('--exp_name', type=str, default='experiment', help="Experiment name")
    parser.add_argument('-N', type=int, default=5, help="Number of arms")
    parser.add_argument('-S', type=int, default=4, help="Number of states in each arm (when applicable, e.g., SIS)")
    parser.add_argument('-A', type=int, default=2, help="Number of actions in each arm (not currently implemented)")
    parser.add_argument('-B', type=float, default=1.0, help="Budget per round")
    parser.add_argument('--reward_bound', type=int, default=1, help="Rescale rewards to this value (only some environments)")

    parser.add_argument('--agent_steps', type=int, default=10, help="Number of rollout steps between epochs")
    parser.add_argument('--agent_epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--agent_init_lambda_trains', type=int, default=0, help="Deprecated, leave at 0")
    parser.add_argument('--agent_clip_ratio', type=float, default=2.0, help="Clip ratio for PPO step")
    parser.add_argument('--agent_final_train_lambdas', type=int, default=10, help="Number of epochs at the end of training to update the policy and critic network, but not the lambda-network")
    parser.add_argument('--agent_start_entropy_coeff', type=float, default=0.0, help="Start entropy coefficient for the cooling procedure")
    parser.add_argument('--agent_end_entropy_coeff', type=float, default=0.0, help="End entropy coefficient for the cooling procedure")
    parser.add_argument('--agent_pi_lr', type=float, default=2e-3, help="Learning rate for policy network")
    parser.add_argument('--agent_vf_lr', type=float, default=2e-3, help="Learning rate for critic network")
    parser.add_argument('--agent_lm_lr', type=float, default=2e-3, help="Learning rate for lambda network")
    parser.add_argument('--agent_train_pi_iters', type=int, default=20, help="Training iterations to run per epoch")
    parser.add_argument('--agent_train_vf_iters', type=int, default=20, help="Training iterations to run per epoch")
    parser.add_argument('--agent_lamb_update_freq', type=int, default=4, help="Number of epochs that should pass before updating the lambda network (so really it is a period, not frequency)")
    


    parser.add_argument('--nature_steps', type=int, default=100, help="Number of rollout steps between epochs")
    parser.add_argument('--nature_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--nature_init_lambda_trains', type=int, help="Deprecated, leave at 0")
    parser.add_argument('--nature_clip_ratio', type=float, default=2.0, help="Clip ratio for PPO step")
    parser.add_argument('--nature_final_train_lambdas', type=int, default=10, help="Number of epochs at the end of training to update the policy and critic networks, but not the lambda-network")
    parser.add_argument('--nature_start_entropy_coeff', type=float, default=0.0, help="Start entropy coefficient for the cooling procedure")
    parser.add_argument('--nature_end_entropy_coeff', type=float, default=0.0, help="End entropy coefficient for the cooling procedure")
    parser.add_argument('--nature_pi_lr_A', type=float, default=1e-3, help="Learning rate for policy network of agent A")
    parser.add_argument('--nature_vf_lr_A', type=float, default=1e-3, help="Learning rate for critic network of agent A")
    parser.add_argument('--nature_pi_lr_B', type=float, default=5e-3, help="Learning rate for policy network of agent B")
    parser.add_argument('--nature_vf_lr_B', type=float, default=5e-3, help="Learning rate for critic network of agent B")
    parser.add_argument('--nature_lm_lr', type=float, default=2e-3, help="Learning rate for lambda network")
    parser.add_argument('--nature_train_pi_iters', type=int, default=20, help="Training iterations to run per epoch")
    parser.add_argument('--nature_train_vf_iters', type=int, default=20, help="Training iterations to run per epoch")
    parser.add_argument('--nature_lamb_update_freq', type=int, default=4, help="Number of epochs that should pass before updating the lambda network (so really it is a period, not frequency)")
    parser.add_argument('--no_hawkins', type=int, default=0, help="If set, will not run Hawkins baselines")

    parser.add_argument('--home_dir', type=str, default='.', help="Home directory for experiments")
    parser.add_argument('--cannon', type=int, default=0, help="Flag used for running experiments on batched slurm-based HPC resources. Leave at 0 for small experiments.")
    parser.add_argument('--n_perturb', type=int, default=3, help="Number of copies of the RLvMid baseline that should be trained/compared against. Each copy will train against a small perturbation of a mid nature strategy.")
    parser.add_argument('--perturbation_size', type=float, default=0.1, help="Size of the perturbation in the above framework")
    parser.add_argument('--pop_size', type=int, default=10, help="If --data==sis, then this sets the population size")
    parser.add_argument('--save_string', type=str, default="exp", help='unique string for saving files related to this experiment')
    parser.add_argument('-d', '--data', default='armman', type=str, help='Environment selection',
                        choices=[   
                                    'random',
                                    'random_reset',
                                    'circulant', 
                                    'armman',
                                    'counterexample',
                                    'sis'
                                ])
    args = parser.parse_args()

    if not args.no_hawkins:
        from robust_rmab.baselines.agent_baselines import HawkinsAgentPolicy


    agent_kwargs = {}
    agent_kwargs['steps_per_epoch'] = args.agent_steps
    agent_kwargs['epochs'] = args.agent_epochs
    agent_kwargs['init_lambda_trains'] = args.agent_init_lambda_trains
    agent_kwargs['clip_ratio'] = args.agent_clip_ratio
    agent_kwargs['final_train_lambdas'] = args.agent_final_train_lambdas
    agent_kwargs['start_entropy_coeff'] = args.agent_start_entropy_coeff
    agent_kwargs['end_entropy_coeff'] = args.agent_end_entropy_coeff
    agent_kwargs['pi_lr'] = args.agent_pi_lr
    agent_kwargs['vf_lr'] = args.agent_vf_lr
    agent_kwargs['lm_lr'] = args.agent_lm_lr
    agent_kwargs['train_pi_iters'] = args.agent_train_pi_iters
    agent_kwargs['train_v_iters'] = args.agent_train_vf_iters
    agent_kwargs['lamb_update_freq'] = args.agent_lamb_update_freq
    agent_kwargs['ac_kwargs'] = dict(hidden_sizes=[args.hid]*args.l)
    agent_kwargs['gamma'] = args.gamma
    



    nature_kwargs = {}
    nature_kwargs['steps_per_epoch'] = args.nature_steps
    nature_kwargs['epochs'] = args.nature_epochs
    nature_kwargs['init_lambda_trains'] = args.nature_init_lambda_trains
    nature_kwargs['clip_ratio'] = args.nature_clip_ratio
    nature_kwargs['final_train_lambdas'] = args.nature_final_train_lambdas
    nature_kwargs['start_entropy_coeff'] = args.nature_start_entropy_coeff
    nature_kwargs['end_entropy_coeff'] = args.nature_end_entropy_coeff
    nature_kwargs['pi_lr_agent'] = args.nature_pi_lr_A
    nature_kwargs['pi_lr_nature'] = args.nature_pi_lr_B
    nature_kwargs['vf_lr_agent'] = args.nature_vf_lr_A
    nature_kwargs['vf_lr_nature'] = args.nature_vf_lr_B
    nature_kwargs['lm_lr'] = args.nature_lm_lr
    nature_kwargs['train_pi_iters'] = args.nature_train_pi_iters
    nature_kwargs['train_v_iters'] = args.nature_train_vf_iters
    nature_kwargs['lamb_update_freq'] = args.nature_lamb_update_freq
    nature_kwargs['ac_kwargs'] = dict(hidden_sizes=[args.hid]*args.l)
    nature_kwargs['gamma'] = args.gamma

    horizon = args.horizon#int(args.agent_steps/args.cpu)
    N=args.N
    S=args.S
    A=args.A
    budget=args.B
    seed=args.seed
    reward_bound=args.reward_bound
    home_dir=args.home_dir
    exp_name=args.exp_name
    max_epochs_double_oracle=args.max_epochs_double_oracle
    gamma = args.gamma

    n_perturb = args.n_perturb

    torch.manual_seed(seed)
    np.random.seed(seed)



    start_time = time.time()

    one_hot_encode = True
    non_ohe_obs_dim = None
    state_norm = 1

    if args.data == 'counterexample':
        from robust_rmab.baselines.nature_baselines_counterexample import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy, DetermNaturePolicy
                        )
    if args.data == 'sis':
        from robust_rmab.baselines.nature_baselines_sis import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy
                        )
        one_hot_encode = False
        non_ohe_obs_dim = 1
        state_norm = args.pop_size



    do = DoubleOracle(data=args.data, N=N, budget=budget, horizon=horizon, 
                    max_epochs_double_oracle=max_epochs_double_oracle,
                    S=S, A=A, seed=seed, reward_bound=reward_bound,
                    home_dir=home_dir, exp_name=exp_name, gamma=gamma,
                    n_simu_epochs=args.n_simu_epochs,
                    agent_kwargs=agent_kwargs, nature_kwargs=nature_kwargs,
                    pop_size=args.pop_size, one_hot_encode=one_hot_encode, state_norm=state_norm,
                             non_ohe_obs_dim=non_ohe_obs_dim)



    # print('max_epochs {}, n_train def {}, nature {}'.format(max_epochs, def_n_train, nature_n_train))
    # print('n_targets {}, horizon {}, budget {}'.format(do.n_targets, horizon, budget))

    if ((not args.data == 'sis') or args.pop_size < 100) and (not args.no_hawkins):
        hawkins_ind = 'pess'
        # baseline_hawkins_pess_agent_i = len(do.agent_strategies)
        add_to_seed = 0#max_epochs_double_oracle + 0 + 3*n_perturb
        pess_nature_params = None
        if args.data == 'counterexample' or args.data=='armman':
            pess_nature_params = do.sampled_nature_parameter_ranges.min(axis=-1)
        elif args.data == 'sis':
            pess_nat = PessimisticNaturePolicy(do.sampled_nature_parameter_ranges, -1)
            pess_nature_params = pess_nat.param_setting
        T = do.env.get_T_for_a_nature(pess_nature_params)
        baseline_hawkins_pess_agent = HawkinsAgentPolicy(N, T, do.env.R, do.env.C, budget, gamma, hawkins_ind)
        do.update_payoffs_agent(baseline_hawkins_pess_agent)

        hawkins_ind = 'mid'
        # baseline_hawkins_middle_agent_i = len(do.agent_strategies)
        add_to_seed = 0#max_epochs_double_oracle + 1 + 3*n_perturb
        if args.data == 'counterexample' or args.data=='armman':
            middle_nature_params = do.sampled_nature_parameter_ranges.mean(axis=-1)
        elif args.data == 'sis':
            mid_nat = MiddleNaturePolicy(do.sampled_nature_parameter_ranges, -1)
            middle_nature_params = mid_nat.param_setting
        T = do.env.get_T_for_a_nature(middle_nature_params)
        baseline_hawkins_middle_agent = HawkinsAgentPolicy(N, T, do.env.R, do.env.C, budget, gamma, hawkins_ind)
        do.update_payoffs_agent(baseline_hawkins_middle_agent)

        hawkins_ind = 'opt'
        # baseline_hawkins_optimist_agent_i = len(do.agent_strategies)
        add_to_seed = 0#max_epochs_double_oracle + 2 + 3*n_perturb
        if args.data == 'counterexample' or args.data=='armman':
            optimist_nature_params = do.sampled_nature_parameter_ranges.max(axis=-1)
        elif args.data == 'sis':
            optimist_nat = OptimisticNaturePolicy(do.sampled_nature_parameter_ranges, -1)
            optimist_nature_params = optimist_nat.param_setting
        T = do.env.get_T_for_a_nature(optimist_nature_params)
        baseline_hawkins_optimist_agent = HawkinsAgentPolicy(N, T, do.env.R, do.env.C, budget, gamma, hawkins_ind)
        do.update_payoffs_agent(baseline_hawkins_optimist_agent)
    


    state_dim = do.env.T.shape[1]

    middle_nature_pol = MiddleNaturePolicy(do.sampled_nature_parameter_ranges, 0)
    do.update_payoffs_nature(middle_nature_pol)

    optimist_nature_pol = OptimisticNaturePolicy(do.sampled_nature_parameter_ranges, 0)
    do.update_payoffs_nature(optimist_nature_pol)




    print('running Double Oracle')
    print('Agent strategies', do.agent_strategies)
    print('Nature strategies', do.nature_strategies)

    agent_eq, nature_eq = do.run()

    print('\n\n\n\n\n-----------------------')
    print('running Double Oracle')
    print('Agent strategies', do.agent_strategies)
    print('Nature strategies', do.nature_strategies)
    # print('equilibrium value is ', val_upper, val_lower)
    print('defender BR mixed strategy           ', np.round(agent_eq, 4))
    print('Nature attractiveness mixed strategy ', np.round(nature_eq, 4))
    # print('Nature attractiveness are')
    # for nature_strategy in do.nature_strategies:
    #     a = convert_to_a(nature_strategy, do.param_int)
    #     print('   ', np.round(a, 3))

    print()
    print('payoffs (regret)', np.array(do.payoffs).shape)
    regret = np.array(do.payoffs) - np.array(do.payoffs).max(axis=0)
    for p in regret:
        print('   ', np.round(p, 2))


    ###########
    # Now compare against baselines -- but need to also compute their nature best responses
    # We will include the new baselines in the final regret calculations, in case any baselines
    # outperform the existing models (shouldn't happen in theory, but makes for a more fair comparison)
    ###########

    n_baseline_comparisons = 0

    # Plan against middle nature
    baseline_middle_rl_i = len(do.agent_strategies)
    for i in range(n_perturb):
        add_to_seed = 0#max_epochs_double_oracle + i + 0*n_perturb
        
        # we want to replicate the shape except for the last dimension
        shape = do.env.sampled_parameter_ranges.shape[:-1]
        
        perturbations = np.random.rand(*shape)

        baseline_middle_rl = use_middle(do, add_to_seed, i, perturbations=perturbations, perturbation_size=args.perturbation_size)
        nature_br = do.nature_oracle.best_response([baseline_middle_rl], [1.0], add_to_seed)
        add_to_seed = 0 # keep this zero so everyone gets the same seeds for the n_simu_epochs
        do.update_payoffs(nature_br, baseline_middle_rl, add_to_seed)
        n_baseline_comparisons+=1

    # Plan against random nature
    # baseline_rl_against_random_i = len(do.agent_strategies)
    # for i in range(n_perturb):
    #     add_to_seed = max_epochs_double_oracle + i + 1*n_perturb
    #     rl_against_random_policy = use_random(do, add_to_seed, i)
    #     do.update_payoffs_agent(rl_against_random_policy)


    # TODO: Use Random agent policy
    baseline_random_agent_i = len(do.agent_strategies)
    # for i in range(n_perturb):
    i=0
    add_to_seed = 0#max_epochs_double_oracle + i + 2*n_perturb
    baseline_random_agent = RandomAgentPolicy(do.env, i)
    nature_br = do.nature_oracle.best_response([baseline_random_agent], [1.0], add_to_seed)
    add_to_seed = 0 # keep this zero so everyone gets the same seeds for the n_simu_epochs
    do.update_payoffs(nature_br, baseline_random_agent, add_to_seed)
    n_baseline_comparisons+=1

    # compute hawkins regrets
    # can't compare against hawkins if state size gets too large because of query time
    if ((not args.data == 'sis') or args.pop_size < 100) and (not args.no_hawkins):
        hawkins_ind = 0
        baseline_hawkins_pess_agent_i = len(do.agent_strategies)
        add_to_seed = 0#max_epochs_double_oracle + 0 + 3*n_perturb
        if args.data == 'counterexample' or args.data=='armman':
            pess_nature_params = do.sampled_nature_parameter_ranges.min(axis=-1)
        elif args.data == 'sis':
            pess_nat = PessimisticNaturePolicy(do.sampled_nature_parameter_ranges, -1)
            pess_nature_params = pess_nat.param_setting
        T = do.env.get_T_for_a_nature(pess_nature_params)
        baseline_hawkins_pess_agent = HawkinsAgentPolicy(N, T, do.env.R, do.env.C, budget, gamma, hawkins_ind)
        nature_br = do.nature_oracle.best_response([baseline_hawkins_pess_agent], [1.0], add_to_seed)
        add_to_seed = 0 # keep this zero so everyone gets the same seeds for the n_simu_epochs
        do.update_payoffs(nature_br, baseline_hawkins_pess_agent, add_to_seed)
        n_baseline_comparisons+=1


        hawkins_ind = 1
        baseline_hawkins_middle_agent_i = len(do.agent_strategies)
        add_to_seed = 0#max_epochs_double_oracle + 1 + 3*n_perturb
        if args.data == 'counterexample' or args.data=='armman':
            middle_nature_params = do.sampled_nature_parameter_ranges.mean(axis=-1)
        elif args.data == 'sis':
            mid_nat = MiddleNaturePolicy(do.sampled_nature_parameter_ranges, -1)
            middle_nature_params = mid_nat.param_setting
        T = do.env.get_T_for_a_nature(middle_nature_params)
        baseline_hawkins_middle_agent = HawkinsAgentPolicy(N, T, do.env.R, do.env.C, budget, gamma, hawkins_ind)
        nature_br = do.nature_oracle.best_response([baseline_hawkins_middle_agent], [1.0], add_to_seed)
        add_to_seed = 0 # keep this zero so everyone gets the same seeds for the n_simu_epochs
        do.update_payoffs(nature_br, baseline_hawkins_middle_agent, add_to_seed)
        n_baseline_comparisons+=1

        hawkins_ind = 2
        baseline_hawkins_optimist_agent_i = len(do.agent_strategies)
        add_to_seed = 0#max_epochs_double_oracle + 2 + 3*n_perturb
        if args.data == 'counterexample' or args.data=='armman':
            optimist_nature_params = do.sampled_nature_parameter_ranges.max(axis=-1)
        elif args.data == 'sis':
            optimist_nat = OptimisticNaturePolicy(do.sampled_nature_parameter_ranges, -1)
            optimist_nature_params = optimist_nat.param_setting
        T = do.env.get_T_for_a_nature(optimist_nature_params)
        baseline_hawkins_optimist_agent = HawkinsAgentPolicy(N, T, do.env.R, do.env.C, budget, gamma, hawkins_ind)
        nature_br = do.nature_oracle.best_response([baseline_hawkins_optimist_agent], [1.0], add_to_seed)
        add_to_seed = 0 # keep this zero so everyone gets the same seeds for the n_simu_epochs
        do.update_payoffs(nature_br, baseline_hawkins_optimist_agent, add_to_seed)
        n_baseline_comparisons+=1



    now = datetime.now()
    str_time = now.strftime('%d-%m-%Y_%H:%M:%S')


    print('After baselines...')
    print('payoffs (regret)', np.array(do.payoffs).shape)
    regret = np.array(do.payoffs) - np.array(do.payoffs).max(axis=0)
    for p in regret:
        print('   ', np.round(p, 2))

    do_regret_no_new_nature = -get_payoff(regret[:-n_baseline_comparisons, :-n_baseline_comparisons], agent_eq, nature_eq)

    df = pd.DataFrame(np.array(agent_eq).reshape(1,-1), columns=list(map(str, do.agent_strategies[:-n_baseline_comparisons])))
    file_name = os.path.join(args.home_dir, 'logs/equilibriums/agent_eq_before_{}_n{}_b{}_h{}_epoch{}_data{}_p{}_s{}.csv'.format(args.save_string, do.N, budget, horizon, max_epochs_double_oracle, args.data, args.pop_size, args.seed))
    df.to_csv(file_name, index=False)
    df = pd.DataFrame(np.array(nature_eq).reshape(1,-1), columns=list(map(str, do.nature_strategies[:-n_baseline_comparisons])))
    file_name = os.path.join(args.home_dir, 'logs/equilibriums/nature_eq_before_{}_n{}_b{}_h{}_epoch{}_data{}_p{}_s{}.csv'.format(args.save_string, do.N, budget, horizon, max_epochs_double_oracle, args.data, args.pop_size, args.seed))
    df.to_csv(file_name, index=False)


    print('old agent equilibrium    ', np.round(agent_eq, 3))
    print('old nature equilibrium ', np.round(nature_eq, 3))
    
    # Compute new equilibrium in case it could increase regret, but don't use baseline agent policies
    agent_eq, nature_eq = do.find_equilibrium_with_regret_array(regret, ignore_rows=n_baseline_comparisons)

    print('new agent equilibrium    ', np.round(agent_eq, 3))
    print('new nature equilibrium ', np.round(nature_eq, 3))

    # compute regret but without the baseline policies
    do_regret = -get_payoff(regret[:-n_baseline_comparisons], agent_eq, nature_eq)

    df = pd.DataFrame(np.array(agent_eq).reshape(1,-1), columns=list(map(str, do.agent_strategies[:-n_baseline_comparisons])))
    file_name = os.path.join(args.home_dir, 'logs/equilibriums/agent_eq_after_{}_n{}_b{}_h{}_epoch{}_data{}_p{}_s{}.csv'.format(args.save_string, do.N, budget, horizon, max_epochs_double_oracle, args.data, args.pop_size, args.seed))
    df.to_csv(file_name, index=False)
    df = pd.DataFrame(np.array(nature_eq).reshape(1,-1), columns=list(map(str, do.nature_strategies)))
    file_name = os.path.join(args.home_dir, 'logs/equilibriums/nature_eq_after_{}_n{}_b{}_h{}_epoch{}_data{}_p{}_s{}.csv'.format(args.save_string, do.N, budget, horizon, max_epochs_double_oracle, args.data, args.pop_size, args.seed))
    df.to_csv(file_name, index=False)


    # compute statistics about the differences between nature policies
    if N <= 5 and args.data=='armman':
        nature_strat_policy_arrays = []
        for i in range(len(do.nature_strategies)):
            state_dim = do.env.T.shape[1]
            nspa, _ = do.nature_strategies[i].get_policy_array(state_dim=state_dim)
            nature_strat_policy_arrays.append(nspa)

        policy_diffs = np.zeros((len(do.nature_strategies), len(do.nature_strategies)))

        for i in range(len(nature_strat_policy_arrays)):
            for j in range(i+1,len(nature_strat_policy_arrays)):
                ns1 = do.nature_strategies[i]
                ns2 = do.nature_strategies[j]

                nspa1 = nature_strat_policy_arrays[i]
                nspa2 = nature_strat_policy_arrays[j]
                
                policy_diffs[i,j] = compute_mean_l2_diff(nspa1, nspa2)
        # print(policy_diffs)

        df = pd.DataFrame(policy_diffs, columns=list(map(str, do.nature_strategies)), index=list(map(str, do.nature_strategies)))
        file_name = os.path.join(args.home_dir, 'logs/nature_pol_diffs/nature_policy_diffs_{}_n{}_b{}_h{}_epoch{}_data{}_p{}_s{}.csv'.format(args.save_string, do.N, budget, horizon, max_epochs_double_oracle, args.data, args.pop_size, args.seed))
        df.to_csv(file_name)



    baseline_middle_rl_regrets = np.zeros(n_perturb)
    for i in range(n_perturb):
        baseline_middle_rl_distrib = np.zeros(len(do.agent_strategies))
        baseline_middle_rl_distrib[baseline_middle_rl_i+i] = 1
        baseline_middle_rl_regrets[i] = do.compute_payoff_regret(baseline_middle_rl_distrib)
    baseline_middle_rl_regret = np.min(baseline_middle_rl_regrets)
    
    print('avg regret of baseline middle {:.3f}'.format(baseline_middle_rl_regret))
    print('regret of all middle baselines', baseline_middle_rl_regrets)


    baseline_random_agent_regrets = np.zeros(1)#np.zeros(n_perturb)
    
    i=0
    baseline_random_agent_distrib = np.zeros(len(do.agent_strategies))
    baseline_random_agent_distrib[baseline_random_agent_i+i] = 1
    baseline_random_agent_regrets[i] = do.compute_payoff_regret(baseline_random_agent_distrib)
    baseline_random_agent_regret = np.min(baseline_random_agent_regrets)
    print('avg regret of baseline random agent {:.3f}'.format(baseline_random_agent_regret))

    if ((not args.data == 'sis') or args.pop_size < 100) and (not args.no_hawkins):
        baseline_hawkins_pess_agent_distrib = np.zeros(len(do.agent_strategies))
        baseline_hawkins_pess_agent_distrib[baseline_hawkins_pess_agent_i] = 1
        baseline_hawkins_pess_agent_regret = do.compute_payoff_regret(baseline_hawkins_pess_agent_distrib)

        baseline_hawkins_middle_agent_distrib = np.zeros(len(do.agent_strategies))
        baseline_hawkins_middle_agent_distrib[baseline_hawkins_middle_agent_i] = 1
        baseline_hawkins_middle_agent_regret = do.compute_payoff_regret(baseline_hawkins_middle_agent_distrib)

        baseline_hawkins_optimist_agent_distrib = np.zeros(len(do.agent_strategies))
        baseline_hawkins_optimist_agent_distrib[baseline_hawkins_optimist_agent_i] = 1
        baseline_hawkins_optimist_agent_regret = do.compute_payoff_regret(baseline_hawkins_optimist_agent_distrib)
    else:
        baseline_hawkins_pess_agent_regret = 0
        baseline_hawkins_middle_agent_regret = 0
        baseline_hawkins_optimist_agent_regret = 0


    
    # print('avg reward of DO {:.3f}, regret {:.3f}'.format(do_rew, do_regret))
    print('avg regret of DO before new nature pols {:.3f}'.format(do_regret_no_new_nature))
    print('avg regret of DO {:.3f}'.format(do_regret))


    print('runtime {:.1f} seconds'.format(time.time() - start_time))

    print('max_epochs {}, n_train def {}, nature {}'.format(max_epochs_double_oracle, args.agent_epochs, args.nature_epochs))
    print('n_targets {}, horizon {}, budget {}'.format(do.N, horizon, budget))

    bar_vals = [
                    do_regret_no_new_nature, 
                    do_regret, 
                    baseline_middle_rl_regret, 

                    baseline_random_agent_regret,
                    baseline_hawkins_pess_agent_regret,
                    baseline_hawkins_middle_agent_regret,
                    baseline_hawkins_optimist_agent_regret
                ]
    tick_names = (
                    'Double Oracle',
                    'Double Oracle+New Nats', 
                    'RL vs. Middle', 

                    'Random Agent',
                    'Hawkins Pessimist Agent',
                    'Hawkins Middle Agent',
                    'Hawkins Optimist Agent'
                  )

    df = pd.DataFrame(np.array(bar_vals).reshape(1,-1), columns=tick_names)
    file_name = os.path.join(args.home_dir, 'logs/regrets/regret_{}_n{}_b{}_h{}_epoch{}_data{}_p{}_s{}.csv'.format(args.save_string, do.N, budget, horizon, max_epochs_double_oracle, args.data, args.pop_size, args.seed))
    df.to_csv(file_name, index=False)

    print('regrets', tick_names)
    print(np.round(bar_vals, 3))

    

    if args.cannon:
        import matplotlib
        matplotlib.use('pdf')

    x = np.arange(len(bar_vals))
    plt.figure()
    plt.bar(x, bar_vals)
    plt.xticks(x, tick_names,rotation=45)
    plt.xlabel('Method')
    plt.ylabel('Mean regret')
    plt.title('N {}, budget {}, horizon {}, max_epochs {}, data {}'.format(do.N, budget, horizon, max_epochs_double_oracle, args.data))
    plt.tight_layout()
    plt.savefig(os.path.join(args.home_dir, 'img/regret_{}_n{}_b{}_h{}_epoch{}_data{}_p{}_s{}.pdf'.format(args.save_string, do.N, budget, horizon, max_epochs_double_oracle, args.data, args.pop_size, args.seed)))
    if not args.cannon:
        plt.show()
