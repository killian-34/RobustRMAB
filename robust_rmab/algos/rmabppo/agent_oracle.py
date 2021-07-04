# Authors: Jackson A. Killian, Lily Xu, 3 July, 2021
# 
# Adapted from repository by: OpenAI
#    - https://spinningup.openai.com/en/latest/



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import torch
from torch.optim import Adam, SGD
import time
import robust_rmab.algos.rmabppo.rmabppo_core as core
from robust_rmab.utils.logx import EpochLogger
from robust_rmab.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from robust_rmab.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from bandit_env import RandomBanditEnv, Eng1BanditEnv, RandomBanditResetEnv, CirculantDynamicsEnv, ARMMANEnv
from bandit_env_robust import ToyRobustEnv, ARMMANRobustEnv, CounterExampleRobustEnv, SISRobustEnv



class RMABPPO_Buffer:
    """
    A buffer for storing trajectories experienced by a RMABPPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, N, act_type, size, one_hot_encode=True, gamma=0.99, lam_OTHER=0.95):
        self.N = N
        self.obs_dim = obs_dim

        self.one_hot_encode = one_hot_encode

        self.obs_buf = np.zeros(core.combined_shape(size, N), dtype=np.float32)
        self.ohs_buf = np.zeros(core.combined_shape(size, (N, obs_dim)), dtype=np.float32)
        
        self.act_buf = np.zeros((size, N), dtype=np.float32)
        self.oha_buf = np.zeros(core.combined_shape(size, (N, act_dim)), dtype=np.float32)

        self.adv_buf = np.zeros((size,N), dtype=np.float32)
        self.rew_buf = np.zeros((size,N), dtype=np.float32)
        self.cost_buf = np.zeros((size,N), dtype=np.float32)
        self.ret_buf = np.zeros((size,N), dtype=np.float32)
        self.val_buf = np.zeros((size,N), dtype=np.float32)
        self.q_buf   = np.zeros((size,N), dtype=np.float32)
        self.logp_buf = np.zeros((size,N), dtype=np.float32)
        self.cdcost_buf = np.zeros(size, dtype=np.float32)
        self.lamb_buf = np.zeros(size, dtype=np.float32)

        self.gamma, self.lam_OTHER = gamma, lam_OTHER
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.act_type = act_type
        self.act_dim = act_dim


    def store(self, obs, act, rew, cost, val, q, lamb, logp):
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


        self.act_buf[self.ptr] = act
        oha = np.zeros((self.N, self.act_dim))
        for i in range(self.N):
            oha[i, int(act[i])] = 1
        self.oha_buf[self.ptr] = oha

        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.val_buf[self.ptr] = val
        self.q_buf[self.ptr]   = q
        self.lamb_buf[self.ptr] = lamb
        self.logp_buf[self.ptr] = logp
        self.ptr += 1


    # TODO: implement last costs rollout if we create a training procedure that 
    # uses every step as a sample to update lambda
    # for now, we only use the first sample of epoch to update lambda, so "future" costs don't matter much
    def finish_path(self, last_vals=0, last_costs=0):
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
            rews = np.append(self.rew_buf[path_slice, i], last_vals[i])
            # TODO implement training that makes use of last_costs, i.e., use all samples to update lam
            costs = np.append(self.cost_buf[path_slice, i], 0)
            # print(costs)
            lambds = np.append(self.lamb_buf[path_slice], 0)

            arm_summed_costs += costs
            # adjust based on action costs

            rews = rews - lambds*costs

            vals = np.append(self.val_buf[path_slice, i], last_vals[i])
            
            # the next two lines implement GAE-Lambda advantage calculation
            qs = rews[:-1] + self.gamma * vals[1:]
            deltas = qs - vals[:-1]
            self.adv_buf[path_slice, i] = core.discount_cumsum(deltas, self.gamma * self.lam_OTHER)
            
            # the next line computes rewards-to-go, to be targets for the value function
            self.ret_buf[path_slice, i] = core.discount_cumsum(rews, self.gamma)[:-1]

            # store the learned q functions
            self.q_buf[path_slice, i]   = qs
            
            self.path_start_idx = self.ptr


        # the next line computes costs-to-go, to be part of the loss for the lambda net
        self.cdcost_buf[path_slice] = core.discount_cumsum(arm_summed_costs, self.gamma)[:-1]




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
            adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf[:, i])
            self.adv_buf[:, i] = (self.adv_buf[:, i] - adv_mean) / adv_std
        
        
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                adv=self.adv_buf, logp=self.logp_buf, qs=self.q_buf, oha=self.oha_buf, 
                ohs=self.ohs_buf, costs=self.cdcost_buf, lambdas=self.lamb_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


class AgentOracle:

    def __init__(self, data, N, S, A, B, seed, REWARD_BOUND, agent_kwargs=dict(),
        home_dir="", exp_name="", sampled_nature_parameter_ranges=None, robust_keyword="",
        pop_size=0, one_hot_encode=True, non_ohe_obs_dim=None, state_norm=None):

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
        self.robust_keyword = robust_keyword

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

        self.actor_critic=core.MLPActorCriticRMAB
        self.agent_kwargs=agent_kwargs

        self.strat_ind = 0

        # this won't work if we go back to MPI, but doing it now to simplify seeding
        self.env = self.env_fn()
        self.env.seed(seed)
        self.env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges


    # Todo - figure out parallelization with MPI -- not clear how to do this yet, so restrict to single cpu
    def best_response(self, nature_strats, nature_eq, add_to_seed):

        self.strat_ind += 1

        # mpi_fork(args.cpu, is_cannon=args.cannon)  # run parallel code with mpi

        from robust_rmab.utils.run_utils import setup_logger_kwargs

        exp_name = '%s_n%ib%.1fd%sr%sp%s'%(self.exp_name, self.N, self.B, self.data, self.robust_keyword, self.pop_size)
        data_dir = os.path.join(self.home_dir, 'data')
        logger_kwargs = setup_logger_kwargs(exp_name, self.seed, data_dir=data_dir)
        # logger_kwargs = setup_logger_kwargs(self.exp_name, self.seed+add_to_seed, data_dir=data_dir)

        return self.best_response_per_cpu(nature_strats, nature_eq, add_to_seed, seed=self.seed,  logger_kwargs=logger_kwargs, **self.agent_kwargs)

    # add_to_seed is obsolete
    def best_response_per_cpu(self, nature_strats, nature_eq, add_to_seed, actor_critic=core.MLPActorCriticRMAB, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
            vf_lr=1e-3, qf_lr=1e-3, lm_lr=5e-2, train_pi_iters=80, train_v_iters=80, train_q_iters=80,
            lam_OTHER=0.97,
            start_entropy_coeff=0.0, end_entropy_coeff=0.0,
            max_ep_len=1000,
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
        # env = self.env_fn()
        env = self.env
        
        # env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges
        obs_dim = env.observation_space.shape

        

        # Create actor-critic module
        ac = actor_critic(env.observation_space, env.action_space,
            N = env.N, C = env.C, B = env.B, strat_ind=self.strat_ind,
            one_hot_encode = self.one_hot_encode, non_ohe_obs_dim = self.non_ohe_obs_dim,
            state_norm=self.state_norm,
            **ac_kwargs)

        act_dim = ac.act_dim
        obs_dim = ac.obs_dim

        # Sync params across processes
        sync_params(ac)

        # Set up experience buffer
        local_steps_per_epoch = int(steps_per_epoch / num_procs())


        buf = RMABPPO_Buffer(obs_dim, act_dim, env.N, ac.act_type, local_steps_per_epoch,
                        one_hot_encode=self.one_hot_encode, gamma=gamma, lam_OTHER=lam_OTHER)

        FINAL_TRAIN_LAMBDAS = final_train_lambdas


        # Set up function for computing RMABPPO policy loss
        def compute_loss_pi(data, entropy_coeff):
            ohs, act, adv, logp_old, lambdas, obs = data['ohs'], data['act'], data['adv'], data['logp'], data['lambdas'], data['obs']

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
                pi, logp = ac.pi_list[i](full_obs[:, i], act[:, i])
                ent = pi.entropy().mean()
                ratio = torch.exp(logp - logp_old[:, i])
                clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv[:, i]
                loss_pi = -(torch.min(ratio * adv[:, i], clip_adv)).mean()
                
                # subtract entropy term since we want to encourage it 
                loss_pi -= entropy_coeff*ent


                loss_pi_list[i] = loss_pi

                # Useful extra info
                approx_kl = (logp_old[:, i] - logp).mean().item()
                
                clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
                clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
                pi_info = dict(kl=approx_kl, ent=ent.item(), cf=clipfrac)
                pi_info_list[i] = pi_info

            return loss_pi_list, pi_info_list

        # Set up function for computing value loss
        def compute_loss_v(data):
            ohs, ret, lambdas, obs = data['ohs'], data['ret'], data['lambdas'], data['obs']
            lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
            full_obs = None
            if ac.one_hot_encode:
                full_obs = torch.cat([ohs, lamb_to_concat], axis=2)
            else:
                obs = obs/self.state_norm
                obs = obs.reshape(obs.shape[0], obs.shape[1], 1)
                full_obs = torch.cat([obs, lamb_to_concat], axis=2)

            loss_list = np.zeros(env.N,dtype=object)
            for i in range(env.N):
                loss_list[i] = ((ac.v_list[i](full_obs[:, i]) - ret[:, i])**2).mean()
            return loss_list

        def compute_loss_q(data):

            ohs, qs, oha, lambdas  = data['ohs'], data['qs'], data['oha'], data['lambdas']
            lamb_to_concat = np.repeat(lambdas, env.N).reshape(-1,env.N,1)
            full_obs = None
            if ac.one_hot_encode:
                full_obs = torch.cat([ohs, lamb_to_concat], axis=2)
            else:
                obs = obs/self.state_norm
                obs = obs.reshape(obs.shape[0], obs.shape[1], 1)
                full_obs = torch.cat([obs, lamb_to_concat], axis=2)

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
            # print('term 1', , 'cost',disc_cost)
            # print('term 1',env.B/(1-gamma))
            # print('cost',disc_cost)

            loss = lamb*(env.B/(1-gamma) - disc_cost)
            print(loss)

            return loss

        # Set up optimizers for policy and value function
        pi_optimizers = np.zeros(env.N,dtype=object)
        vf_optimizers = np.zeros(env.N,dtype=object)
        qf_optimizers = np.zeros(env.N,dtype=object)

        for i in range(env.N):
            pi_optimizers[i] = Adam(ac.pi_list[i].parameters(), lr=pi_lr)
            # pi_optimizers[i] = SGD(ac.pi_list[i].parameters(), lr=pi_lr)
            vf_optimizers[i] = Adam(ac.v_list[i].parameters(), lr=vf_lr)
            # vf_optimizers[i] = SGD(ac.v_list[i].parameters(), lr=vf_lr)
            qf_optimizers[i] = Adam(ac.q_list[i].parameters(), lr=qf_lr)
            # qf_optimizers[i] = SGD(ac.q_list[i].parameters(), lr=qf_lr)
        # lambda_optimizer = Adam(ac.lambda_net.parameters(), lr=lm_lr)
        lambda_optimizer = SGD(ac.lambda_net.parameters(), lr=lm_lr)


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
                    pi_optimizers[i].zero_grad()
                loss_pi, pi_info = compute_loss_pi(data, entropy_coeff)
                for i in range(env.N):
                    kl = mpi_avg(pi_info[i]['kl'])
                    # if kl > 1.5 * target_kl:
                    #     logger.log('Early stopping at step %d due to reaching max kl.'%i)
                    #     break
                    loss_pi[i].backward()
                    mpi_avg_grads(ac.pi_list[i])    # average grads across MPI processes
                    pi_optimizers[i].step()

            logger.store(StopIter=i)

            # Value function learning
            for i in range(train_v_iters):
                for i in range(env.N):
                    vf_optimizers[i].zero_grad()
                loss_v = compute_loss_v(data)
                for i in range(env.N):
                    loss_v[i].backward()
                    mpi_avg_grads(ac.v_list[i])    # average grads across MPI processes
                    vf_optimizers[i].step()


            # Lambda optimization
            if epoch%lamb_update_freq == 0 and epoch > 0 and (epochs - epoch) > FINAL_TRAIN_LAMBDAS:
                # for i in range(train_lam_iters):

                # Should only update this once because we only get one sample from the environment
                # unless we are running parallel instances
                lambda_optimizer.zero_grad()
                loss_lamb = compute_loss_lambda(data)
                
                loss_lamb.backward()
                last_param = list(ac.lambda_net.parameters())[-1]
                # print('last param',last_param)
                # print('grad',last_param.grad)

                mpi_avg_grads(ac.lambda_net)    # average grads across MPI processes
                lambda_optimizer.step()
                

        # Prepare for interaction with environment
        start_time = time.time()
        current_lamb = 0

        o, ep_actual_ret, ep_lamb_adjusted_ret, ep_len = env.reset(), 0, 0, 0
        o = o.reshape(-1)


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



        # Sample a nature policy
        nature_eq = np.array(nature_eq)
        nature_eq[nature_eq < 0] = 0
        nature_eq = nature_eq / nature_eq.sum()
        print('nature_eq')
        print(nature_eq)
        nature_pol = np.random.choice(nature_strats,p=nature_eq)


        # Main loop: collect experience in env and update/log each epoch
        head_entropy_coeff_schedule = np.linspace(start_entropy_coeff, end_entropy_coeff, epochs)
        for epoch in range(epochs):
            print("start state",o)
            current_lamb = 0
            with torch.no_grad():
                # this is the version where we only predict lambda once at the top of the epoch...
                current_lamb = ac.lambda_net(torch.as_tensor(o, dtype=torch.float32))
                logger.store(Lamb=current_lamb)


            # Resample nature policy every time we update lambda
            if epoch%lamb_update_freq == 0 and epoch > 0:
                nature_pol = np.random.choice(nature_strats,p=nature_eq)


            for t in range(local_steps_per_epoch):
                torch_o = torch.as_tensor(o, dtype=torch.float32)
                a_agent, v, logp, q, probs = ac.step(torch_o, current_lamb)
                a_nature = nature_pol.get_nature_action(torch_o)

                a_nature_env = nature_pol.bound_nature_actions(a_nature, state=o, reshape=True)

                if (local_steps_per_epoch - t) < 25:
                    print('lam',current_lamb,'obs:',o,'a',a_agent,'v:',v,'probs:',probs)

                next_o, r, d, _ = env.step(a_agent, a_nature_env)
                
                next_o = next_o.reshape(-1)
                
                actual_r = r.sum()
                cost_vec = np.zeros(env.N)
                for i in range(env.N):
                    cost_vec[i] = env.C[a_agent[i]]

                lamb_adjusted_r = r.sum() - current_lamb*cost_vec.sum()

                ep_actual_ret += actual_r.sum()
                ep_lamb_adjusted_ret += lamb_adjusted_r
                ep_len += 1




                # save and log
                buf.store(o, a_agent, r, cost_vec, v, q, current_lamb, logp)
                logger.store(VVals=v)
                
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
                        print('lam',current_lamb,'obs:',o,'a',a_agent,'v:',v,'probs:',probs)
                        _, v, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32), current_lamb)

                        # rollout costs for an imagined 50 steps...
                        
                        last_costs = np.zeros((FINAL_ROLLOUT_LENGTH, env.N))
   

                    else:
                        v = 0
                        last_costs = np.zeros((FINAL_ROLLOUT_LENGTH, env.N))
                    buf.finish_path(v, last_costs)

                    # only save EpRet / EpLen if trajectory finished
                    # if terminal:
                    logger.store(EpActualRet=ep_actual_ret, EpLambAdjRet=ep_lamb_adjusted_ret, EpLen=ep_len)
                    o, ep_actual_ret, ep_lamb_adjusted_ret, ep_len = env.reset(), 0, 0, 0
                    o = o.reshape(-1)


            # Save model
            if (epoch == epochs-1):
                print("saving")
                logger.save_state({'env': env}, None)

            # Perform RMABPPO update!
            head_entropy_coeff = head_entropy_coeff_schedule[epoch]
            update(epoch, head_entropy_coeff)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpActualRet', with_min_and_max=True)
            # logger.log_tabular('EpActualRet', average_only=True)
            logger.log_tabular('EpLambAdjRet', with_min_and_max=True)
            # logger.log_tabular('EpLambAdjRet', average_only=True)
            # logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)

            # logger.log_tabular('StopIter', average_only=True)
            logger.log_tabular('Lamb', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


        return ac


    def simulate_reward(self, agent_pol, nature_pol, seed=0, 
            steps_per_epoch=100, epochs=100, gamma=0.99):

        # make a new env for computing returns 
        env = self.env_fn()
        # important to make sure these are always the same for all instatiations of the env
        env.sampled_parameter_ranges = self.sampled_nature_parameter_ranges

        env.seed(seed)

        o, ep_actual_ret, ep_lamb_adjusted_ret, ep_len = env.reset(), 0, 0, 0
        o = o.reshape(-1)
       
        rewards = np.zeros((epochs, steps_per_epoch))
        for epoch in range(epochs):
            # print("epoch",epoch)

            for t in range(steps_per_epoch):
                torch_o = torch.as_tensor(o, dtype=torch.float32)
                a_agent  = agent_pol.act_test(torch_o)
                a_nature = nature_pol.get_nature_action(torch_o)
                a_nature_env = nature_pol.bound_nature_actions(a_nature, state=o, reshape=True)



                next_o, r, d, _ = env.step(a_agent, a_nature_env)
                
                next_o = next_o.reshape(-1)
                
                actual_r = r.sum()

                ep_actual_ret += actual_r
                # ep_lamb_adjusted_ret += lamb_adjusted_r
                ep_len += 1


                rewards[epoch,t] = actual_r*(gamma**t)

                # Update obs (critical!)
                o = next_o



            # loop again
            o, ep_actual_ret, ep_lamb_adjusted_ret, ep_len = env.reset(), 0, 0, 0
            o = o.reshape(-1)



        rewards = rewards.sum(axis=1).mean()
        print("Rewards:", rewards)


        return rewards



# python3 spinup/algos/pytorch/ppo/rmab_rl_lambda_ppo.py --hid 64 -l 2 --gamma 0.9 --cpu 1 --step 100 -N 4 -S 2 -A 2 -B 1 --REWARD_BOUND 2 --exp_name rmab_rl_bandit_n4s2a2b1_r2_lambda -s 0 --epochs 1000 --init_lambda_trains 1
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--hid', type=int, default=64, help="Number of units in each layer of the neural networks used for the Oracles")
    parser.add_argument('-l', type=int, default=2, help="Depth of the neural networks used for Agent and Nature Oracles (i.e., layers)")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--seed', '-s', type=int, default=0, help="Seed")
    parser.add_argument('--cpu', type=int, default=1, help="Number of processes for mpi")
    
    parser.add_argument('--exp_name', type=str, default='experiment', help="Experiment name")
    parser.add_argument('-N', type=int, default=5, help="Number of arms")
    parser.add_argument('-S', type=int, default=4, help="Number of states in each arm (when applicable, e.g., SIS)")
    parser.add_argument('-A', type=int, default=2, help="Number of actions in each arm (not currently implemented)")
    parser.add_argument('-B', type=float, default=1.0, help="Budget per round")
    parser.add_argument('--reward_bound', type=int, default=1, help="Rescale rewards to this value (only some environments)")
    parser.add_argument('--save_string', type=str, default="")

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

    parser.add_argument('--pop_size', type=int, default=0)

    parser.add_argument('--home_dir', type=str, default='.', help="Home directory for experiments")
    parser.add_argument('--cannon', type=int, default=0, help="Flag used for running experiments on batched slurm-based HPC resources. Leave at 0 for small experiments.")
    parser.add_argument('-d', '--data', default='counterexample', type=str, help='Environment selection',
                        choices=[   
                                    'random',
                                    'random_reset',
                                    'circulant', 
                                    'armman',
                                    'counterexample',
                                    'sis'
                                ])

    parser.add_argument('--robust_keyword', default='pess', type=str, help='Method for picking some T out of the uncertain environment',
                        choices=[   
                                    'pess',
                                    'mid',
                                    'opt', # i.e., optimistic
                                    'sample_random'
                                ])

    args = parser.parse_args()

    mpi_fork(args.cpu, is_cannon=args.cannon)  # run parallel code with mpi

    # from spinup.utils.run_utils import setup_logger_kwargs

    # exp_name = '%s_n%is%ia%ib%.2fr%.2f'%(args.exp_name, args.N, args.S, args.A, args.B, args.REWARD_BOUND)
    # print(exp_name)
    # data_dir = os.path.join(args.home_dir, 'data')
    # logger_kwargs = setup_logger_kwargs(exp_name, args.seed, data_dir=data_dir)

    N = args.N
    S = args.S
    A = args.A
    B = args.B
    budget = B
    reward_bound = args.reward_bound
    seed=args.seed
    data = args.data
    home_dir = args.home_dir
    exp_name=args.exp_name
    gamma = args.gamma

    torch.manual_seed(seed)
    np.random.seed(seed)

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

    env_fn = None

    one_hot_encode = True
    non_ohe_obs_dim = None
    state_norm = None

    if args.data == 'counterexample':
        from nature_baselines_counterexample import   (
                    RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                    OptimisticNaturePolicy, DetermNaturePolicy, SampledRandomNaturePolicy
                )
        env_fn = lambda : CounterExampleRobustEnv(N,B,seed)

    if args.data == 'armman':
        from nature_baselines_armman import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy, SampledRandomNaturePolicy
                        )
        env_fn = lambda: ARMMANRobustEnv(N,B,seed)

    if args.data == 'sis':
        from nature_baselines_sis import   (
                            RandomNaturePolicy, PessimisticNaturePolicy, MiddleNaturePolicy, 
                            OptimisticNaturePolicy, SampledRandomNaturePolicy
                        )
        env_fn = lambda: SISRobustEnv(N,B,args.pop_size,seed)
        
        # don't one hot encode this state space...
        one_hot_encode = False
        non_ohe_obs_dim = 1
        state_norm = args.pop_size


    env = env_fn()
    sampled_nature_parameter_ranges = env.sample_parameter_ranges()
    # important to make sure these are always the same for all instatiations of the env
    env.sampled_parameter_ranges = sampled_nature_parameter_ranges

    agent_oracle  = AgentOracle(data, N, S, A, budget, seed, reward_bound,
                             agent_kwargs=agent_kwargs, home_dir=home_dir, exp_name=exp_name,
                             robust_keyword=args.robust_keyword,
                             sampled_nature_parameter_ranges = sampled_nature_parameter_ranges,
                             pop_size=args.pop_size, one_hot_encode=one_hot_encode, state_norm=state_norm,
                             non_ohe_obs_dim=non_ohe_obs_dim)

    nature_strategy = None
    if args.robust_keyword == 'mid':
        nature_strategy = MiddleNaturePolicy(sampled_nature_parameter_ranges, 0)

    if args.robust_keyword == 'sample_random':
        nature_strategy = SampledRandomNaturePolicy(sampled_nature_parameter_ranges, 0)

        # init the random strategy
        nature_strategy.sample_param_setting(seed)



    add_to_seed = 0
    agent_oracle.best_response([nature_strategy], [1.0], add_to_seed)




