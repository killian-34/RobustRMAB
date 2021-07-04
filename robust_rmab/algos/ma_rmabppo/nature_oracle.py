import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
from torch.optim import Adam, SGD
import time
import robust_rmab.algos.ma_rmabppo.ma_rmabppo_core as core
from robust_rmab.utils.logx import EpochLogger
from robust_rmab.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from robust_rmab.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from bandit_env import RandomBanditEnv, Eng1BanditEnv, RandomBanditResetEnv, CirculantDynamicsEnv
from bandit_env_robust import ToyRobustEnv, ARMMANRobustEnv, CounterExampleRobustEnv, SISRobustEnv

import matplotlib as mpl
import matplotlib.pyplot as plt
import os
mpl.use('tkagg')


class MA_RMABPPO_Buffer:
    """
    A buffer for storing trajectories experienced by a MA_RMABPPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim_agent, act_dim_nature, N, act_type, size, one_hot_encode=True, gamma=0.99, lam_OTHER=0.95):
        self.N = N
        self.obs_dim = obs_dim
        self.act_dim_agent = act_dim_agent
        self.act_dim_nature = act_dim_nature
        self.one_hot_encode = one_hot_encode

        self.obs_buf = np.zeros(core.combined_shape(size, N), dtype=np.float32)
        self.ohs_buf = np.zeros(core.combined_shape(size, (N, obs_dim)), dtype=np.float32)
        
        self.act_buf_agent = np.zeros((size, N), dtype=np.float32)
        self.act_buf_nature = np.zeros((size, act_dim_nature), dtype=np.float32)
        # self.oha_buf = np.zeros(core.combined_shape(size, (N, act_dim)), dtype=np.float32)

        self.adv_buf_agent = np.zeros((size,N), dtype=np.float32)
        self.rew_buf_agent = np.zeros((size,N), dtype=np.float32)
        self.cost_buf = np.zeros((size,N), dtype=np.float32)
        self.ret_buf_agent = np.zeros((size,N), dtype=np.float32)
        self.val_buf_agent = np.zeros((size,N), dtype=np.float32)
        self.q_buf_agent   = np.zeros((size,N), dtype=np.float32)
        self.logp_buf_agent = np.zeros((size,N), dtype=np.float32)
        self.cdcost_buf = np.zeros(size, dtype=np.float32)
        self.lamb_buf = np.zeros(size, dtype=np.float32)

        self.adv_buf_nature = np.zeros(size, dtype=np.float32)
        self.rew_buf_nature = np.zeros(size, dtype=np.float32)
        self.ret_buf_nature = np.zeros(size, dtype=np.float32)
        self.val_buf_nature = np.zeros(size, dtype=np.float32)
        self.q_buf_nature   = np.zeros(size, dtype=np.float32)
        self.logp_buf_nature = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam_OTHER = gamma, lam_OTHER
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.act_type = act_type
        


    def store(self, obs, cost, lamb, act_agent, act_nature, rew_agent, rew_nature, 
                    val_agent, val_nature, q_agent, q_nature, logp_agent, logp_nature):
                        
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        ohs = np.zeros((self.N, self.obs_dim))
        if self.one_hot_encode:
            for i in range(self.N):
                ohs[i, int(obs[i])] = 1
        self.ohs_buf[self.ptr] = ohs


        self.act_buf_agent[self.ptr] = act_agent
        self.act_buf_nature[self.ptr] = act_nature
        # oha = np.zeros((self.N, self.act_dim))
        # for i in range(self.N):
        #     oha[i, int(act[i])] = 1
        # self.oha_buf[self.ptr] = oha

        self.rew_buf_agent[self.ptr] = rew_agent
        self.cost_buf[self.ptr] = cost
        self.val_buf_agent[self.ptr] = val_agent
        self.q_buf_agent[self.ptr]   = q_agent
        self.lamb_buf[self.ptr] = lamb
        self.logp_buf_agent[self.ptr] = logp_agent

        self.rew_buf_nature[self.ptr] = rew_nature
        self.val_buf_nature[self.ptr] = val_nature
        self.q_buf_nature[self.ptr]   = q_nature
        self.logp_buf_nature[self.ptr] = logp_nature

        self.ptr += 1


    
    def finish_path(self, last_vals_agent=0, last_costs=0, last_val_nature=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)

        arm_summed_costs = np.zeros(self.ptr - self.path_start_idx + 1)

        for i in range(self.N):
            rews_agent = np.append(self.rew_buf_agent[path_slice, i], last_vals_agent[i])
            # TODO implement training that makes use of last_costs, i.e., use all samples to update lam
            costs = np.append(self.cost_buf[path_slice, i], 0)
            # print(costs)
            lambds = np.append(self.lamb_buf[path_slice], 0)

            arm_summed_costs += costs
            # adjust based on action costs

            rews_agent = rews_agent - lambds*costs

            vals_agent = np.append(self.val_buf_agent[path_slice, i], last_vals_agent[i])
            
            # the next two lines implement GAE-Lambda advantage calculation
            qs_agent = rews_agent[:-1] + self.gamma * vals_agent[1:]
            deltas_agent = qs_agent - vals_agent[:-1]
            self.adv_buf_agent[path_slice, i] = core.discount_cumsum(deltas_agent, self.gamma * self.lam_OTHER)
            
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf_agent[path_slice, i] = core.discount_cumsum(rews_agent, self.gamma)[:-1]

            # store the learned q functions
            self.q_buf_agent[path_slice, i]   = qs_agent
            
            self.path_start_idx = self.ptr


        # the next line computes costs-to-go, to be part of the loss for the lambda net
        self.cdcost_buf[path_slice] = core.discount_cumsum(arm_summed_costs, self.gamma)[:-1]


        rews_nature = np.append(self.rew_buf_nature[path_slice], last_val_nature)
        vals_nature = np.append(self.val_buf_nature[path_slice], last_val_nature)

        qs_nature = rews_nature[:-1] + self.gamma * vals_nature[1:]
        deltas_nature = qs_nature - vals_nature[:-1]

        self.adv_buf_nature[path_slice] = core.discount_cumsum(deltas_nature, self.gamma * self.lam_OTHER)
        self.ret_buf_nature[path_slice] = core.discount_cumsum(rews_nature, self.gamma)[:-1]
        self.q_buf_nature[path_slice]   = qs_nature



    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        for i in range(self.N):
            # the next two lines implement the advantage normalization trick
            adv_mean_agent, adv_std_agent = mpi_statistics_scalar(self.adv_buf_agent[:, i])
            self.adv_buf_agent[:, i] = (self.adv_buf_agent[:, i] - adv_mean_agent) / adv_std_agent
        
        adv_mean_nature, adv_std_nature = mpi_statistics_scalar(self.adv_buf_nature)
        self.adv_buf_nature = (self.adv_buf_nature - adv_mean_nature) / adv_std_nature


        data = dict(obs=self.obs_buf, act_agent=self.act_buf_agent, ret_agent=self.ret_buf_agent,
                adv_agent=self.adv_buf_agent, logp_agent=self.logp_buf_agent, qs_agent=self.q_buf_agent,
                act_nature=self.act_buf_nature, ret_nature=self.ret_buf_nature,
                adv_nature=self.adv_buf_nature, logp_nature=self.logp_buf_nature, qs_nature=self.q_buf_nature,
                ohs=self.ohs_buf, costs=self.cdcost_buf, lambdas=self.lamb_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}



class NatureOracle:

    def __init__(self, data, N, S, A, B, seed, REWARD_BOUND, nature_kwargs=dict(),
        home_dir="", exp_name="", sampled_nature_parameter_ranges=None,
        pop_size=0, one_hot_encode=True, non_ohe_obs_dim=None, state_norm=1,
        nature_state_norm=1):

        self.data = data
        self.home_dir = home_dir
        self.exp_name = exp_name
        self.REWARD_BOUND = REWARD_BOUND
        self.N = N
        self.S = S
        self.A = A
        self.B = B
        self.seed=seed
        self.sampled_nature_parameter_ranges = sampled_nature_parameter_ranges
        self.nature_state_norm = nature_state_norm

        self.pop_size = pop_size
        self.one_hot_encode = one_hot_encode
        self.non_ohe_obs_dim = non_ohe_obs_dim
        self.state_norm = state_norm

        if data == 'random':
            self.env_fn = lambda : RandomBanditEnv(N,S,A,B,seed,REWARD_BOUND)

        if data == 'random_reset':
            self.env_fn = lambda : RandomBanditResetEnv(N,S,A,B,seed,REWARD_BOUND)

        if data == 'armman':
            self.env_fn = lambda : ARMMANRobustEnv(N,B,seed)

        if data == 'circulant':
            self.env_fn = lambda : CirculantDynamicsEnv(N,B,seed)

        if data == 'counterexample':
            self.env_fn = lambda : CounterExampleRobustEnv(N,B,seed)

        if data == 'sis':
            self.env_fn = lambda : SISRobustEnv(N,B,pop_size,seed)

        self.ma_actor_critic = core.RMABLambdaNatureOracle
        self.nature_kwargs=nature_kwargs

        self.strat_ind = -1

        # this won't work if we go back to MPI, but doing it now to simplify seeding
        self.env = self.env_fn()
        self.env.seed(seed)
        self.env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges


    # Todo - figure out parallelization with MPI -- not clear how to do this yet, so restrict to single cpu
    def best_response(self, nature_strats, nature_eq, add_to_seed):

        self.strat_ind+=1

        # mpi_fork(args.cpu, is_cannon=args.cannon)  # run parallel code with mpi

        from robust_rmab.utils.run_utils import setup_logger_kwargs

        exp_name = '%s_n%is%ia%ib%.2fr%.2f'%(self.exp_name, self.N, self.S, self.A, self.B, self.REWARD_BOUND)
        data_dir = os.path.join(self.home_dir, 'data')
        logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed, data_dir=data_dir)
        # logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed+add_to_seed, data_dir=data_dir)

        return self.best_response_per_cpu(nature_strats, nature_eq, add_to_seed, seed=self.seed, logger_kwargs=logger_kwargs, **self.nature_kwargs)

    # add_to_seed is obsolete
    def best_response_per_cpu(self, agent_strats, agent_eq, add_to_seed, ma_actor_critic=core.RMABLambdaNatureOracle, ac_kwargs=dict(), 
        seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr_agent=3e-4, pi_lr_nature=3e-4,
        vf_lr_agent=1e-3, vf_lr_nature=1e-3, qf_lr=1e-3, lm_lr=5e-2, 
        train_pi_iters=80, train_v_iters=80, train_q_iters=80,
        lam_OTHER=0.97, max_ep_len=1000,
        start_entropy_coeff=0.0, end_entropy_coeff=0.0,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10,
        lamb_update_freq=10,
        init_lambda_trains=0,
        final_train_lambdas=0):
        
        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        # Set up logger and save configuration
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())


        # Instantiate environment

        env = self.env
        
        obs_dim = env.observation_dimension
        action_dim_nature = env.action_dim_nature


        # Create actor-critic module
        ac = ma_actor_critic(env.observation_space, env.action_space, env.sampled_parameter_ranges,
             env.action_dim_nature, env=env,
             N = env.N, C = env.C, B = env.B, strat_ind = self.strat_ind,
             one_hot_encode = self.one_hot_encode, non_ohe_obs_dim = self.non_ohe_obs_dim,
            state_norm=self.state_norm, nature_state_norm=self.nature_state_norm,
            **ac_kwargs)

        act_dim_agent = ac.act_dim_agent
        act_dim_nature = ac.act_dim_nature
        obs_dim = ac.obs_dim

        # Sync params across processes
        sync_params(ac)


        # Set up experience buffer
        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        buf = MA_RMABPPO_Buffer(obs_dim, act_dim_agent, act_dim_nature, env.N, ac.act_type, local_steps_per_epoch, 
            one_hot_encode=self.one_hot_encode, gamma=gamma, lam_OTHER=lam_OTHER)

        FINAL_TRAIN_LAMBDAS = final_train_lambdas


        # Set up function for computing MA_RMABPPO policy loss
        def compute_loss_pi_agent(data, entropy_coeff):
            ohs, act, adv, logp_old, lambdas, obs = data['ohs'], data['act_agent'], data['adv_agent'], data['logp_agent'], data['lambdas'], data['obs']

            lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
            full_obs = None
            if ac.one_hot_encode:
                full_obs = torch.cat([ohs, lamb_to_concat], axis=2)
            else:
                obs = obs/self.state_norm
                obs = obs.reshape(obs.shape[0], obs.shape[1], 1)
                full_obs = torch.cat([obs, lamb_to_concat], axis=2)

            loss_pi_list = np.zeros(env.N,dtype=object)
            pi_info_list = np.zeros(env.N,dtype=object)

            # Policy loss
            for i in range(env.N):
                pi, logp = ac.pi_list_agent[i](full_obs[:, i], act[:, i])
                ent = pi.entropy().mean()
                ratio = torch.exp(logp - logp_old[:, i])
                clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv[:, i]
                loss_pi = -(torch.min(ratio * adv[:, i], clip_adv)).mean()
                
                # subtract entropy term since we want to encourage it 
                loss_pi -= entropy_coeff*ent

                loss_pi_list[i] = loss_pi

                # Useful extra info
                approx_kl = (logp_old[:, i] - logp).mean().item()
                # ent = pi.entropy().mean().item()
                clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
                clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                pi_info = dict(kl=approx_kl, ent=ent.item(), cf=clipfrac)
                pi_info_list[i] = pi_info

            return loss_pi_list, pi_info_list

        # Set up function for computing MA_RMABPPO policy loss
        def compute_loss_pi_nature(data, entropy_coeff):
            obs, act, adv, logp_old = data['obs'], data['act_nature'], data['adv_nature'], data['logp_nature']

            if not ac.one_hot_encode:
                obs = obs/self.nature_state_norm

            # Policy loss
            pi, logp = ac.pi_nature(obs, act)
            ent = pi.entropy().mean()
            ratio = torch.exp(logp - logp_old)

            clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv

            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            # subtract entropy term since we want to encourage it 
            loss_pi -= entropy_coeff*ent


            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

            return loss_pi, pi_info

        # Set up function for computing value loss
        def compute_loss_v_agent(data):
            ohs, ret, lambdas, act_nature, obs = data['ohs'], data['ret_agent'], data['lambdas'], data['act_nature'], data['obs']

            # some semi-annoying array manip to broadcast the single value of lambda
            # to fit the data shape of the observations
            lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
            
            a_nature_env = np.zeros(act_nature.shape)

            # print(act_nature)
            for i in range(len(act_nature)):
                a_nature_env[i] = ac.bound_nature_actions(act_nature[i], state=obs[i], reshape=False)

            # Similar semi-annoying array manip
            a_nature_env = np.repeat(a_nature_env,env.N,axis=0).reshape(-1, env.N, a_nature_env.shape[1])

            if ac.one_hot_encode:
                full_obs = torch.cat([ohs, lamb_to_concat, torch.as_tensor(a_nature_env, dtype=torch.float32)], axis=2)
            else:
                obs = obs/self.state_norm
                obs = obs.reshape(obs.shape[0], obs.shape[1], 1)
                full_obs = torch.cat([obs, lamb_to_concat, torch.as_tensor(a_nature_env, dtype=torch.float32)], axis=2)

            
            # full_obs = torch.cat([ohs, lamb_to_concat, a_nature_env], axis=2)

            loss_list = np.zeros(env.N,dtype=object)
            for i in range(env.N):
                loss_list[i] = ((ac.v_list_agent[i](full_obs[:, i]) - ret[:, i])**2).mean()
            return loss_list


        # nature value function takes agent action as input
        def compute_loss_v_nature(data):
            obs, ret, act_agent = data['obs'], data['ret_nature'], data['act_agent']
            # oha_agent = np.zeros(ac.act_dim_agent)
            # oha_agent[int(act_agent)] = 1
            if not self.one_hot_encode:
                obs = obs/self.nature_state_norm

            x_s_a_agent = torch.as_tensor(np.concatenate([obs, act_agent],axis=1), dtype=torch.float32)
            return ((ac.v_nature(x_s_a_agent) - ret)**2).mean()


        # Deprecated
        def compute_loss_q(data):

            ohs, qs, oha, lambdas  = data['ohs'], data['qs'], data['oha'], data['lambdas']
            lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
            full_obs = torch.cat([ohs, lamb_to_concat], axis=2)

            loss_list = np.zeros(env.N,dtype=object)
            for i in range(env.N):
                x = torch.as_tensor(np.concatenate([full_obs[:, i], oha[:, i]], axis=1), dtype=torch.float32)
                loss_list[i] = ((ac.q_list[i](x) - qs[:, i])**2).mean()
            return loss_list


        def compute_loss_lambda(data):

            disc_cost = data['costs'][0]
            # lamb = data['lambdas'][0]
            obs = data['obs'][0]
            if not self.one_hot_encode:
                obs = obs/self.state_norm

            lamb = ac.lambda_net(torch.as_tensor(obs,dtype=torch.float32))
            print('lamb',lamb, 'term 1', env.B/(1-gamma), 'cost',disc_cost, 'diff', env.B/(1-gamma) - disc_cost)
            
            loss = lamb*(env.B/(1-gamma) - disc_cost)
            print(loss)

            return loss

        # Set up optimizers for policy and value function
        pi_agent_optimizers = np.zeros(env.N,dtype=object)
        vf_agent_optimizers = np.zeros(env.N,dtype=object)
        qf_nature_optimizers = np.zeros(env.N,dtype=object)

        for i in range(env.N):
            pi_agent_optimizers[i] = Adam(ac.pi_list_agent[i].parameters(), lr=pi_lr_agent)
            # pi_optimizers[i] = SGD(ac.pi_list[i].parameters(), lr=pi_lr)
            vf_agent_optimizers[i] = Adam(ac.v_list_agent[i].parameters(), lr=vf_lr_agent)
            # vf_optimizers[i] = SGD(ac.v_list[i].parameters(), lr=vf_lr)
            qf_nature_optimizers[i] = Adam(ac.q_list_agent[i].parameters(), lr=qf_lr)
            # qf_optimizers[i] = SGD(ac.q_list[i].parameters(), lr=qf_lr)
        # lambda_optimizer = Adam(ac.lambda_net.parameters(), lr=lm_lr)
        lambda_optimizer = SGD(ac.lambda_net.parameters(), lr=lm_lr)
        pi_nature_optimizer = Adam(ac.pi_nature.parameters(), lr=pi_lr_nature)
        # pi_optimizers[i] = SGD(ac.pi_list[i].parameters(), lr=pi_lr)
        vf_nature_optimizer = Adam(ac.v_nature.parameters(), lr=vf_lr_nature)
        # vf_optimizers[i] = SGD(ac.v_list[i].parameters(), lr=vf_lr)
        qf_nature_optimizer = Adam(ac.q_nature.parameters(), lr=qf_lr)


        # Set up model saving
        logger.setup_pytorch_saver(ac)

        def update(epoch, head_entropy_coeff):
            data = buf.get()

            entropy_coeff = 0.0
            if (epochs - epoch) > FINAL_TRAIN_LAMBDAS:
                # cool entropy down as we relearn policy for each lambda
                entropy_coeff_schedule = np.linspace(head_entropy_coeff,0,lamb_update_freq)
                # don't rotate
                # entropy_coeff_schedule = entropy_coeff_schedule[1:] + entropy_coeff_schedule[:1]
                ind = epoch%lamb_update_freq
                entropy_coeff = entropy_coeff_schedule[ind]
            print('entropy',entropy_coeff)


            # Train policy with multiple steps of gradient descent
            for i in range(train_pi_iters):
                for i in range(env.N):
                    pi_agent_optimizers[i].zero_grad()
                loss_pi_agent, pi_info_agent = compute_loss_pi_agent(data, entropy_coeff)
                for i in range(env.N):
                    kl = mpi_avg(pi_info_agent[i]['kl'])
                    # if kl > 1.5 * target_kl:
                    #     logger.log('Early stopping at step %d due to reaching max kl.'%i)
                    #     break
                    loss_pi_agent[i].backward()
                    mpi_avg_grads(ac.pi_list_agent[i])    # average grads across MPI processes
                    pi_agent_optimizers[i].step()


            logger.store(StopIter=i)



            # Value function learning
            for i in range(train_v_iters):
                for i in range(env.N):
                    vf_agent_optimizers[i].zero_grad()
                loss_v_agent = compute_loss_v_agent(data)
                for i in range(env.N):
                    loss_v_agent[i].backward()
                    mpi_avg_grads(ac.v_list_agent[i])    # average grads across MPI processes
                    vf_agent_optimizers[i].step()

                


            # Lambda optimization
            # sync nature updates with lambda updates..
            # But Stop training lambdas after a certain point
            if epoch%lamb_update_freq == 0 and epoch > 0:
                # for i in range(train_lam_iters):

                # Should only update this once because we only get one sample from the environment
                # unless we are running parallel instances
                # also, eventually freze lambda training
                if (epochs - epoch) > FINAL_TRAIN_LAMBDAS:
                    lambda_optimizer.zero_grad()
                    loss_lamb = compute_loss_lambda(data)
                    
                    loss_lamb.backward()
                    last_param = list(ac.lambda_net.parameters())[-1]
                    # print('last param',last_param)
                    # print('grad',last_param.grad)

                    mpi_avg_grads(ac.lambda_net)    # average grads across MPI processes
                    lambda_optimizer.step()


                # UPDATE the nature policy
                entropy_coeff = 0.0

                # Train policy with multiple steps of gradient descent
                for i in range(train_pi_iters):
                    pi_nature_optimizer.zero_grad()
                    loss_pi_nature, pi_info_nature = compute_loss_pi_nature(data, entropy_coeff)
                    kl = mpi_avg(pi_info_nature['kl'])
                    # if kl > 1.5 * target_kl:
                    #     logger.log('Early stopping at step %d due to reaching max kl.'%i)
                    #     break
                    loss_pi_nature.backward()
                    mpi_avg_grads(ac.pi_nature)    # average grads across MPI processes
                    pi_nature_optimizer.step()


                for i in range(train_v_iters):

                    vf_nature_optimizer.zero_grad()
                    loss_v_nature = compute_loss_v_nature(data)
                    loss_v_nature.backward()
                    mpi_avg_grads(ac.v_nature)    # average grads across MPI processes
                    vf_nature_optimizer.step()




        # Prepare for interaction with environment
        start_time = time.time()
        current_lamb = 0


        o, ep_actual_ret_agent, ep_actual_ret_nature, ep_lamb_adjusted_ret_agent, ep_lamb_adjusted_ret_nature, ep_len = env.reset(), 0, 0, 0, 0, 0
        o = o.reshape(-1)

        init_o = np.copy(o)

        losses = {'pi_agent': [], 'v_agent': [], 'pi_nature': [], 'v_nature': [],
         'r_agent_lam':[], 'r_nature_lam':[],
         'r_agent':[], 'r_nature':[],
         'epoch_lams':[],
         'a_nature_0_01':[], 'a_nature_1_01':[],
         'a_agent_prob_01':[], 'step_lams_01':[],
         'a_nature_0_10':[], 'a_nature_1_10':[],
         'a_agent_prob_10':[], 'step_lams_10':[],
         'a_nature_0_11':[], 'a_nature_1_11':[],
         'a_agent_prob_11':[], 'step_lams_11':[],

         }


        INIT_LAMBDA_TRAINS = init_lambda_trains

        # Initialize lambda to make large predictions
        for i in range(INIT_LAMBDA_TRAINS):
            init_lambda_optimizer = SGD(ac.lambda_net.parameters(), lr=lm_lr)
            init_lambda_optimizer.zero_grad()
            loss_lamb = ac.return_large_lambda_loss(o, gamma)
            
            loss_lamb.backward()
            last_param = list(ac.lambda_net.parameters())[-1]

            mpi_avg_grads(ac.lambda_net)    # average grads across MPI processes
            init_lambda_optimizer.step()


        # always act on arm in state 0
        def get_action_test_policy(obs):
            a = np.zeros(obs.shape[0])
            if int(obs[0])==0 and int(obs[1])==0:
                choice = np.random.choice(np.arange(2))
                a[choice] = 1
            elif int(obs[0])==1 and int(obs[1])==1:
                # choice = np.random.choice(np.arange(2))
                # a[choice] = 1
                pass
            elif int(obs[0]==0):
                choice = 0
                a[choice] = 1
            elif int(obs[1]==0):
                choice = 1
                a[choice] = 1
            return a


        NUM_TEST_POLICY_RUNS = 50
        # Main loop: collect experience in env and update/log each epoch

        # Sample an agent policy
        
        # sometimes get negative values that are tiny e.g., -6.54284594e-18, just set them to 0
        agent_eq = np.array(agent_eq)
        agent_eq[agent_eq < 0] = 0
        agent_eq = agent_eq / agent_eq.sum()
        print('agent_eq')
        print(agent_eq)

        agent_pol = np.random.choice(agent_strats, p=agent_eq)

        head_entropy_coeff_schedule = np.linspace(start_entropy_coeff, end_entropy_coeff, epochs)
        for epoch in range(epochs):

            
            env.current_full_state = init_o
            o = init_o


            print("start state",o)
            current_lamb = 0
            with torch.no_grad():
                current_lamb = ac.lambda_net(torch.as_tensor(o, dtype=torch.float32))
                logger.store(Lamb=current_lamb)

            
            # Resample agent policy every time we update lambda
            if epoch%lamb_update_freq == 0 and epoch > 0:
                agent_pol = np.random.choice(agent_strats, p=agent_eq)


            for t in range(local_steps_per_epoch):

                torch_o = torch.as_tensor(o, dtype=torch.float32)
                a_agent, v_agent, logp_agent, q_agent, a_nature, v_nature, logp_nature, q_nature, probs_agent = ac.step(torch_o, current_lamb)

                a_nature_env = ac.bound_nature_actions(a_nature, state=o, reshape=True)

                next_o, r_agent, d, _ = env.step(a_agent, a_nature_env)
                next_o = next_o.reshape(-1)
            

                s = time.time()
                test_r_list = np.zeros(NUM_TEST_POLICY_RUNS)

                # only need to sample the action once if this is deterministic
                a_test = agent_pol.act_test(torch_o)
                for trial in range(NUM_TEST_POLICY_RUNS):
                    env.current_full_state = o
                    # uncomment this if you want stochastic actions from the agent
                    # a_test = agent_pol.act_test(torch_o)
                    
                    next_o_sample, r_test, _, _ = env.step(a_test, a_nature_env)

                    test_r_list[trial] = r_test.sum()
                endt = time.time()

                print('Time sampling:',(endt-s))

                env.current_full_state = next_o


                r_test_mean = test_r_list.mean()


                actual_r_agent = r_agent.sum()
                actual_r_nature = actual_r_agent - r_test_mean

                cost_vec = np.zeros(env.N)
                for i in range(env.N):
                    cost_vec[i] = env.C[a_agent[i]]

                # only using this reward for debugging training
                # we will store and manipulate raw rewards during training, i.e., r_agent
                lamb_adjusted_r_agent = actual_r_agent - current_lamb*cost_vec.sum()

                # but store lambda adjusted for nature oracle...
                lamb_adjusted_r_nature = actual_r_nature - current_lamb*cost_vec.sum()

                ep_actual_ret_agent += actual_r_agent
                ep_lamb_adjusted_ret_agent += lamb_adjusted_r_agent

                ep_actual_ret_nature += actual_r_nature
                ep_lamb_adjusted_ret_nature += lamb_adjusted_r_nature

                ep_len += 1

                # save and log
                buf.store(o, cost_vec, current_lamb, a_agent, a_nature, r_agent, lamb_adjusted_r_nature, v_agent, v_nature,
                    q_agent, q_nature, logp_agent, logp_nature)
                logger.store(VVals_agent=v_agent)
                logger.store(VVals_nature=v_nature)


                
                # Update obs (critical!)
                o = next_o

                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t==local_steps_per_epoch-1

                if terminal or epoch_ended:
                    FINAL_ROLLOUT_LENGTH = 50
                    if epoch_ended and not(terminal):
                        # print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                        pass
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        print('lam',current_lamb,'obs:',o,'a_agent',a_agent,'v_agent:',v_agent,
                             'logp_agent:',logp_agent,'a_nature',a_nature,'v_nature:',v_nature,
                             'logp_nature:',logp_nature)
                        _, v_agent, _, _, _, v_nature, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32), current_lamb)

                        # rollout costs for an imagined 50 steps...
                        
                        last_costs = np.zeros((FINAL_ROLLOUT_LENGTH, env.N))
                        
                    else:
                        v_agent = 0
                        v_nature = 0
                        last_costs = np.zeros((FINAL_ROLLOUT_LENGTH, env.N))
                    buf.finish_path(v_agent, last_costs, v_nature)

                    # only save EpRet / EpLen if trajectory finished
                    # if terminal:
                    logger.store(EpActualRetAgent=ep_actual_ret_agent, EpLambAdjRetAgent=ep_lamb_adjusted_ret_agent,
                                    EpLRetNature=ep_actual_ret_nature, EpLambAdjRetNature=ep_lamb_adjusted_ret_nature, EpLen=ep_len)

                    losses['r_agent_lam'].append(ep_lamb_adjusted_ret_agent)
                    losses['r_nature_lam'].append(ep_lamb_adjusted_ret_nature)

                    losses['r_agent'].append(ep_actual_ret_agent)
                    losses['r_nature'].append(ep_actual_ret_nature)

                    losses['epoch_lams'].append(current_lamb)

                    o, ep_actual_ret_agent, ep_actual_ret_nature, ep_lamb_adjusted_ret_agent, ep_lamb_adjusted_ret_nature, ep_len = env.reset(), 0, 0, 0, 0, 0
                    o = o.reshape(-1)
                    init_o = np.copy(o)


            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                logger.save_state({'env': env}, None)

            # Perform MA_RMABPPO update!
            head_entropy_coeff = head_entropy_coeff_schedule[epoch]
            update(epoch, head_entropy_coeff)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpActualRetAgent', with_min_and_max=True)
            # logger.log_tabular('EpActualRet', average_only=True)
            logger.log_tabular('EpLambAdjRetAgent', with_min_and_max=True)
            logger.log_tabular('EpLambAdjRetNature', with_min_and_max=True)
            # logger.log_tabular('EpLambAdjRet', average_only=True)
            # logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals_agent', with_min_and_max=True)
            logger.log_tabular('VVals_nature', with_min_and_max=True)
            logger.log_tabular('Lamb', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

        return ac



# __main__ is deprecated

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=64)
#     parser.add_argument('-l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--cpu', type=int, default=4)
#     parser.add_argument('--steps', type=int, default=4000)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='ppo')
#     parser.add_argument('-N', type=int, default=4)
#     parser.add_argument('-S', type=int, default=4)
#     parser.add_argument('-A', type=int, default=2)
#     parser.add_argument('-B', type=float, default=1.0)
#     parser.add_argument('--REWARD_BOUND', type=int, default=1)
#     parser.add_argument('--init_lambda_trains', type=int, default=1)
#     parser.add_argument('--clip_ratio', type=float, default=2.0)#, default=0.2)
#     parser.add_argument('--final_train_lambdas', type=int, default=10)
#     parser.add_argument('--start_entropy_coeff', type=float, default=0.0)
#     parser.add_argument('--end_entropy_coeff', type=float, default=0.0)
#     args = parser.parse_args()

#     mpi_fork(args.cpu)  # run parallel code with mpi

#     from spinup.utils.run_utils import setup_logger_kwargs

#     exp_name = '%s_n%is%ia%ib%.2fr%.2f'%(args.exp_name, args.N, args.S, args.A, args.B, args.REWARD_BOUND)
#     print(exp_name)
#     logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

#     N = args.N
#     S = args.S
#     A = args.A
#     B = args.B
#     REWARD_BOUND = args.REWARD_BOUND

#     mappo(lambda : ToyRobustEnv(N,B,args.seed), ma_actor_critic=core.RMABLambdaNatureOracle,
#         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
#         seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
#         logger_kwargs=logger_kwargs,
#         pi_lr_agent=1e-3,
#         vf_lr_agent=1e-3, 
#         pi_lr_nature=5e-3,
#         vf_lr_nature=5e-3, 
#         qf_lr=1e-3,
#         lm_lr=2e-3,
#         start_entropy_coeff=args.start_entropy_coeff,
#         end_entropy_coeff=args.end_entropy_coeff,
#         train_pi_iters=10, train_v_iters=10, train_q_iters=5,
#         lamb_update_freq=4,
#         init_lambda_trains=args.init_lambda_trains,
#         clip_ratio=args.clip_ratio,
#         final_train_lambdas=args.final_train_lambdas)




