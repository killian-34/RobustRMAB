import numpy as np
import gym



class RandomBanditEnv(gym.Env):
    def __init__(self, N, S, A, B, seed, REWARD_BOUND):


        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.observation_dimension = 1
        self.action_dimension = 1
        self.REWARD_BOUND = REWARD_BOUND
        self.reward_range = (0, REWARD_BOUND)
        self.S = S
        self.A = A
        self.B = B
        self.init_seed = seed

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        self.seed(seed=seed)
        self.T, self.R, self.C = self.get_experiment()




    def get_experiment(self):
        print('getting random')
        def random_T(S,A):
            T = self.random_stream.dirichlet(np.ones(S), size=(S,A))
            return T

        T = np.zeros((self.N,self.S,self.A,self.S))
        for i in range(self.N):
            T[i] = random_T(self.S,self.A)

        # R = np.sort(self.random_stream.rand(self.N, self.S), axis=1)*self.REWARD_BOUND
        R = np.array([np.arange(self.S) for _ in range(self.N)])*self.REWARD_BOUND/(self.S-1)


        # C = np.concatenate([[0], np.sort(np.random.rand(self.A-1))])
        C = np.arange(self.A)



        return T, R, C



    def step(self, a):

        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=int(self.current_full_state[i])
            next_arm_state=np.argmax(self.random_stream.multinomial(1, self.T[i, current_arm_state, int(a[i]), :]))
            next_full_state[i]=next_arm_state
            rewards[i] = self.R[i, next_arm_state]

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None

    def reset(self):
        self.current_full_state = np.zeros(self.N, dtype=int)
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]





class RandomBanditResetEnv(gym.Env):
    def __init__(self, N, S, A, B, seed, REWARD_BOUND):


        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.observation_dimension = 1
        self.action_dimension = 1
        self.REWARD_BOUND = REWARD_BOUND
        self.reward_range = (0, REWARD_BOUND)
        self.S = S
        self.A = A
        self.B = B
        self.init_seed = seed

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        self.seed(seed=seed)
        self.T, self.R, self.C = self.get_experiment()




    def get_experiment(self):
        print('getting random')
        def random_T(S,A):
            T = self.random_stream.dirichlet(np.ones(S), size=(S,A))
            return T

        T = np.zeros((self.N,self.S,self.A,self.S))
        for i in range(self.N):
            T[i] = random_T(self.S,self.A)

        # R = np.sort(self.random_stream.rand(self.N, self.S), axis=1)*self.REWARD_BOUND
        R = np.array([np.arange(self.S) for _ in range(self.N)])*self.REWARD_BOUND/(self.S-1)


        # C = np.concatenate([[0], np.sort(np.random.rand(self.A-1))])
        C = np.arange(self.A)



        return T, R, C



    def step(self, a):

        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=int(self.current_full_state[i])
            next_arm_state=np.argmax(self.random_stream.multinomial(1, self.T[i, current_arm_state, int(a[i]), :]))
            next_full_state[i]=next_arm_state
            rewards[i] = self.R[i, next_arm_state]

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None

    def reset(self):
        # self.current_full_state = np.zeros(self.N, dtype=int)
        self.current_full_state = self.random_stream.choice(self.observation_space, self.N)
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]






class Eng1BanditEnv(RandomBanditEnv):
    def __init__(self, N, S, A, B, seed, REWARD_BOUND):

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.REWARD_BOUND = REWARD_BOUND
        self.reward_range = (0, REWARD_BOUND)
        self.S = S
        self.A = A
        self.B = B

        self.init_seed = seed

        self.current_full_state = np.zeros(N)

        self.random_stream = np.random.RandomState()
        self.seed(seed)

        self.T, self.R, self.C = self.get_experiment()


    def get_experiment(self):

        def eng1_T(S, A):

            T = np.zeros((S,A,S))

            prior_weight = 5
            for i in range(S):
                for j in range(A):
                    prior = np.ones(S)
                    add_vector = np.zeros(S)
                    add_vector[i]+= prior_weight*abs(j-A)
                    prior += add_vector
                    T[i,j] = self.random_stream.dirichlet(prior)
            return T

        T = np.zeros((self.N,self.S,self.A,self.S))
        for i in range(self.N):
            T[i] = eng1_T(self.S,self.A)


        R = np.array([np.arange(self.S) for _ in range(self.N)])*self.REWARD_BOUND/(self.S-1)


        # C = np.array([0, 1, 3, 6])
        C = np.arange(self.A)


        return T, R, C



# Continuous state and action example!

class SISBanditEnv(gym.Env):
    def __init__(self, N, population_sizes, B, seed, init_infection_size, REWARD_BOUND):


        self.N = N
        self.observation_space = 2 # because SIS model has two compartments
        self.num_params = 2 # SIS model has two params
        self.action_space = 1 # because we will only be able to modify one parameter

        self.observation_dimension = 2 # because SIS model has two compartments
        self.action_dimension = 1 # because we will only be able to modify one parameter

        self.REWARD_BOUND = REWARD_BOUND
        self.reward_range = (0, REWARD_BOUND)
        self.population_sizes = population_sizes
        self.init_infection_size = init_infection_size
        self.B = B
        self.init_seed = seed
        self.epsilon = 1e-4

        self.current_full_state = np.zeros((N,self.observation_space))
        self.random_stream = np.random.RandomState()
        self.seed(seed=seed)
        self.current_full_state = self.reset()

        self.params = self.get_experiment()

        self.T, self.R, self.C = None, None, None


    # SIS model: https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIS_model
    # has params beta and gamma that determine the rate of transition between states
    def get_experiment(self):

        beta_range = [0, 0.8]
        gamma_range = [0, 0.8]

        print('getting random SIS experiment')

        params = np.zeros((self.N, self.num_params))

        def get_sis_params():
            beta = self.random_stream.rand()*beta_range[1] + beta_range[0]
            gamma = self.random_stream.rand()*gamma_range[1] + gamma_range[0]
            return beta, gamma


        for i in range(self.N):
            params[i] = get_sis_params()


        params = [[0.01,1],
         [0.01,1],
         [1,0.5],
         [1,0.5]
        ]



        print(params)


        return params

    def rewards_all(self):

        rewards_all = np.zeros(self.N)
        for i in range(self.N):
            rewards_all[i] = self.reward(self.current_full_state[i])
        return rewards_all

    def reward(self, state):
        # reward is sum of susceptible people
        return state[0]/self.population_sizes[0]


    def costs_all(self, actions_all):
        actions_all = actions_all.reshape(-1)
        # action cost is identity for now
        action_costs = np.zeros(self.N)
        for i in range(self.N):
            action_costs[i] = self.cost(actions_all[i])
        return action_costs

    def cost(self, action):
        # action cost is identity for now
        return action


    # define how the actions work
    def step(self, a):

        def evolve_sis(S, I, N, params):
            new_s = -params[0] * S * I / N + params[1]*I
            new_i = params[0] * S * I / N - params[1]*I
            return S+new_s, I+new_i

        def shape_params_with_action(a,params):
            # we want a=1 ==> beta=\epsilon
            # and a=0 ==> beta=beta
            new_params = np.copy(params)
            beta = params[0]
            new_beta = abs(a*beta - beta)
            new_beta = max(new_beta, self.epsilon)
            new_params[0] = new_beta
            return new_params

        ###### Get next state
        a = a.reshape(-1)
        next_full_state = np.zeros(self.current_full_state.shape)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=self.current_full_state[i]
            arm_action = a[i]
            arm_params = self.params[i]
            a_adjusted_arm_params = shape_params_with_action(arm_action, arm_params)

            next_arm_state=evolve_sis(current_arm_state[0], current_arm_state[1], self.population_sizes[i], a_adjusted_arm_params)

            next_full_state[i]=next_arm_state
            rewards[i] = self.reward(next_arm_state)

        self.current_full_state = next_full_state

        return next_full_state, rewards, False, None

    
    def reset(self):
        self.current_full_state = np.zeros((self.N, self.observation_space))
        self.current_full_state[:,0] = self.init_infection_size
        self.current_full_state[:,1] = self.population_sizes[:] - self.init_infection_size
        return self.current_full_state

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]











class CirculantDynamicsEnv(gym.Env):
    def __init__(self, N, B, seed):#, REWARD_BOUND):


        S = 4
        A = 2

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.observation_dimension = 1
        self.action_dimension = 1
        # self.REWARD_BOUND = REWARD_BOUND
        # self.reward_range = (0, REWARD_BOUND)
        self.S = S
        self.A = A
        self.B = B
        self.init_seed = seed

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        self.seed(seed=seed)
        self.T, self.R, self.C = self.get_experiment(N)




    def get_experiment(self, N):
        T1 = np.array([[[0.5, 0, 0,0.5], #for state 0 action 0
                  [0.5, 0.5, 0, 0]],#for state 0 action 1

                  [[0.5, 0.5, 0, 0],#for state 1 action 0
                  [0, 0.5, 0.5, 0]],#for state 1 action 1

                  [[0, 0.5, 0.5, 0],#for state 2 action 0
                  [0, 0, 0.5, 0.5]],#for state 2 action 1

                  [[0, 0, 0.5, 0.5],#for state 3 action 0
                  [0.5, 0, 0, 0.5]] #for state 3 action 1
                  ])

        T = np.array([T1 for _ in range(N)])

        R = np.array([[-1,0,0,1] for _ in range(N)]) # rewards
        C = np.array([0, 1])

        # prioritize arms in state 2; if none at state 2, pull at state 1; then at state 0, then 3,

        return T, R, C



    def step(self, a):

        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=int(self.current_full_state[i])
            next_arm_state=np.argmax(self.random_stream.multinomial(1, self.T[i, current_arm_state, int(a[i]), :]))
            next_full_state[i]=next_arm_state
            rewards[i] = self.R[i, next_arm_state]

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None

    def reset(self):
        # self.current_full_state = np.zeros(self.N, dtype=int)
        self.current_full_state = self.random_stream.choice(self.observation_space, self.N)
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]




class ARMMANEnv(gym.Env):
    def __init__(self, N, B, seed):#, REWARD_BOUND):


        S = 3
        A = 2

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.observation_dimension = 1
        self.action_dimension = 1
        # self.REWARD_BOUND = REWARD_BOUND
        # self.reward_range = (0, REWARD_BOUND)
        self.S = S
        self.A = A
        self.B = B
        self.init_seed = seed

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        self.seed(seed=seed)
        self.T, self.R, self.C = self.get_experiment(N)


    def get_experiment(self, N):
        
        percent_A = 0.2

        percent_B = 0.2


        # States go S, P, L
        # 

        # A - 10 in A
        tA = np.array([[[0.1, 0.9, 0.0], 
                        [0.1, 0.9, 0.0]],
                        [[0, 0.2, 0.8],
                        [0.8, 0.2, 0]],
                        [[0, 0.4, 0.6],
                        [0.0, 0.4, 0.6]]
                        ])
        # B - 10 in B
        tB = np.array([[[0.9, 0.1, 0.0], 
                        [0.9, 0.1, 0.0]],
                        [[0, 0.6, 0.4],
                        [0.4, 0.6, 0]],
                        [[0, 0.4, 0.6],
                        [0.0, 0.4, 0.6]]
                        ])

        # C - 30 in C
        tC = np.array([[[0.1, 0.9, 0.0], 
                        [0.1, 0.9, 0.0]],
                        [[0, 0.9, 0.1],
                        [0.1, 0.9, 0]],
                        [[0, 0.4, 0.6],
                        [0.0, 0.4, 0.6]]
                        ])

        

        # make the loopy for making the tmatrix
        num_A = int(N*percent_A)
        num_B = int(N*percent_B)
        num_C = N  - num_A - num_B

        T = []
        for i in range(num_A):
            T.append(tA)
        for i in range(num_B):
            T.append(tB)
        for i in range(num_C):
            T.append(tC)

        T = np.array(T)
        R = np.array([[1,0.5,0] for _ in range(N)])
        C = np.array([0, 1])

        # prioritize arms in state 2; if none at state 2, pull at state 1; then at state 0, then 3,

        return T, R, C



    def step(self, a):

        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=int(self.current_full_state[i])
            next_arm_state=np.argmax(self.random_stream.multinomial(1, self.T[i, current_arm_state, int(a[i]), :]))
            next_full_state[i]=next_arm_state
            rewards[i] = self.R[i, next_arm_state]

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None

    def reset(self):
        # self.current_full_state = np.zeros(self.N, dtype=int)
        self.current_full_state = self.random_stream.choice(self.observation_space, self.N)
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]



