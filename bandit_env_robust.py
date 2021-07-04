import numpy as np
import gym
import torch
from scipy.special import comb


class ToyRobustEnv(gym.Env):
    def __init__(self, N, B, seed):

        S = 2
        A = 2

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)

        self.observation_dimension = 1
        self.action_dimension = 1

        self.S = S
        self.A = A
        self.B = B
        self.init_seed = seed

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        self.parameter_ranges = np.array([
                # (0.5, 1.0)      # p11a for each arm
                (0.5, 0.75)      # p11a for each arm
        ]*N)

        self.seed(seed=seed)
        self.T, self.R, self.C = self.get_experiment()




    def get_experiment(self):


        T = np.zeros((self.N,self.S,self.A,self.S))

        T_i = [
            [
                [0.5, 0.5], # p^p_00, p^p_01
                [0.5, 0.5] # p^p_10, p^p_11
            ],
            [
                [0.5, 0.5], # p^a_00, p^a_01
                [-1,   -1] # p^a_10, p^a_11 -- these will be set by the parameter
            ]

        ]

        for i in range(self.N):
            T[i] = T_i

        R = np.array([[0, 1] for _ in range(self.N)])

        C = np.arange(self.A)


        return T, R, C


    # a_agent should correspond to an action respresented in the transition matrix
    # a_nature should be a probability in the range specified by self.parameter_ranges
    def step(self, a_agent, a_nature):

        for i,param in enumerate(a_nature):
            if param < self.parameter_ranges[i][0] or param > self.parameter_ranges[i][1]:
                raise ValueError("nature action outside allowed param range. Was %s but should be in %s"%(param, self.parameter_ranges[i]))
            else:
                # else let nature set the transition prob as appropriate
                self.T[i][1,1,1] = param
                self.T[i][1,1,0] = 1 - param


        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=int(self.current_full_state[i])
            next_arm_state=np.argmax(self.random_stream.multinomial(1, self.T[i, current_arm_state, int(a_agent[i]), :]))
            next_full_state[i]=next_arm_state
            rewards[i] = self.R[i, next_arm_state]

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None

    def reset(self):
        self.current_full_state = np.zeros(self.N, dtype=int)
        # self.current_full_state = self.random_stream.choice(self.observation_space, self.N)
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def reset_random(self):
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






class ARMMANRobustEnv_Old(gym.Env):
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

        self.percent_A = 0.2
        self.percent_B = 0.2

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        self.parameter_ranges = self.get_parameter_ranges(self.N)

        self.seed(seed=seed)
        self.T, self.R, self.C = self.get_experiment(N)



    def get_parameter_ranges(self, N):
        
        # A - 10 in A
        rangeA = [0.3, 0.5]

        # B - 10 in B
        # rangeB = [0.2, 0.8]
        # rangeB = [0.3, 0.6]
        rangeB = [0.1, 0.6]

        # C - 30 in C
        rangeC = [0.2, 0.4]

        

        num_A = int(N*self.percent_A)
        num_B = int(N*self.percent_B)
        num_C = N  - num_A - num_B

        parameter_ranges = []
        for i in range(num_A):
            parameter_ranges.append(rangeA)
        for i in range(num_B):
            parameter_ranges.append(rangeB)
        for i in range(num_C):
            parameter_ranges.append(rangeC)

        # self.parameter_ranges = np.array(parameter_ranges)

        return np.array(parameter_ranges)


    def get_experiment(self, N):
        
        percent_A = 0.2

        percent_B = 0.2


        # States go S, P, L
        # 

        # A - 10 in A
        tA = np.array([[[0.1, 0.9, 0.0], 
                        [0.1, 0.9, 0.0]],

                        [[0, -1, -1],
                        [-1, -1, 0]],

                        [[0, 0.4, 0.6],
                        [0.0, 0.4, 0.6]]
                        ])
        # B - 10 in B
        tB = np.array([[[0.9, 0.1, 0.0], 
                        [0.9, 0.1, 0.0]],
                        [[0, -1, -1],
                        [-1, -1, 0]],
                        [[0, 0.4, 0.6],
                        [0.0, 0.4, 0.6]]
                        ])

        # tB = np.array([[[0.1, 0.9, 0.0], 
        #                 [0.1, 0.9, 0.0]],
        #                 [[0, -1, -1],
        #                 [-1, -1, 0]],
        #                 [[0, 0.4, 0.6],
        #                 [0.0, 0.4, 0.6]]
        #                 ])

        # C - 30 in C
        tC = np.array([[[0.1, 0.9, 0.0], 
                        [0.1, 0.9, 0.0]],
                        [[0, -1, -1],
                        [-1, -1, 0]],
                        [[0, 0.4, 0.6],
                        [0.0, 0.4, 0.6]]
                        ])

        

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


    # env has only binary actions so random is easy to generate
    def random_agent_action(self):
        actions = np.zeros(self.N)
        choices = np.random.choice(np.arange(self.N), int(self.B), replace=False)
        actions[choices] = 1
        return actions


    # a_agent should correspond to an action respresented in the transition matrix
    # a_nature should be a probability in the range specified by self.parameter_ranges
    def step(self, a_agent, a_nature):

        for i,param in enumerate(a_nature):
            if param < self.parameter_ranges[i][0] or param > self.parameter_ranges[i][1]:
                raise ValueError("nature action outside allowed param range. Was %s but should be in %s"%(param, self.parameter_ranges[i]))
            else:
                # else let nature set the transition prob as appropriate
                self.T[i][1,0,2] = param
                self.T[i][1,0,1] = 1-param

                # self.T[i][1,1,0] = param
                # self.T[i][1,1,1] = 1-param

                self.T[i][1,1,0] = 1-param
                self.T[i][1,1,1] = param

        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=int(self.current_full_state[i])
            next_arm_state=np.argmax(self.random_stream.multinomial(1, self.T[i, current_arm_state, int(a_agent[i]), :]))
            next_full_state[i]=next_arm_state
            rewards[i] = self.R[i, next_arm_state]

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None

    # a_agent should correspond to an action respresented in the transition matrix
    # a_nature should be a probability in the range specified by self.parameter_ranges
    def get_T_for_a_nature(self, a_nature):

        for i,param in enumerate(a_nature):
            if param < self.parameter_ranges[i][0] or param > self.parameter_ranges[i][1]:
                raise ValueError("nature action outside allowed param range. Was %s but should be in %s"%(param, self.parameter_ranges[i]))
            else:
                # else let nature set the transition prob as appropriate
                self.T[i][1,0,2] = param
                self.T[i][1,0,1] = 1-param

                self.T[i][1,1,0] = param
                self.T[i][1,1,1] = 1-param

        return np.copy(self.T)

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
            # print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]
















class ARMMANRobustEnv(gym.Env):
    def __init__(self, N, B, seed):#, REWARD_BOUND):


        S = 3
        A = 2

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.observation_dimension = 1
        self.action_dimension = 1
        self.action_dim_nature = N*A
        # self.REWARD_BOUND = REWARD_BOUND
        # self.reward_range = (0, REWARD_BOUND)
        self.S = S
        self.A = A
        self.B = B
        self.init_seed = seed

        self.percent_A = 0.2
        self.percent_B = 0.2

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        self.PARAMETER_RANGES = self.get_parameter_ranges(self.N)

        # make sure to set this whenever environment is created, but do it outside so it always the same
        self.sampled_parameter_ranges = None 


        self.seed(seed=seed)
        self.T, self.R, self.C = self.get_experiment(N)

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()



    # new version has one range per state, per action
    # We will sample ranges from within these to get some extra randomness
    def get_parameter_ranges(self, N):
        
        # A - 10 in A
        rangeA = [
                    [
                        [0.0, 1.0],
                        [0.0, 1.0]
                    ],
                    [
                        [0.5, 1.0], # p deteriorate in absence of intervention
                        [0.5, 1.0], # p improve on intervention
                    ],
                    [
                        [0.35, 0.85],
                        [0.35, 0.85]
                    ]

                ]


        # B - 10 in B
        rangeB = [
                    [
                        [0.0, 1.0],
                        [0.0, 1.0]
                    ],
                    [
                        [0.35, 0.85], # p deteriorate in absence of intervention
                        [0.15, 0.65], # p improve on intervention
                    ],
                    [
                        [0.35, 0.85],
                        [0.35, 0.85]
                    ]

                ]

        # C - 30 in C
        rangeC = [
                    [
                        [0.0, 1.0],
                        [0.0, 1.0]
                    ],
                    [
                        [0.35, 0.85], # p deteriorate in absence of intervention
                        [0.0, 0.5], # p improve on intervention
                    ],
                    [
                        [0.35, 0.85],
                        [0.35, 0.85]
                    ]

                ]

        

        num_A = int(N*self.percent_A)
        num_B = int(N*self.percent_B)
        num_C = N  - num_A - num_B

        parameter_ranges = []
        for i in range(num_A):
            parameter_ranges.append(rangeA)
        for i in range(num_B):
            parameter_ranges.append(rangeB)
        for i in range(num_C):
            parameter_ranges.append(rangeC)

        # self.parameter_ranges = np.array(parameter_ranges)

        return np.array(parameter_ranges)


    def sample_parameter_ranges(self):
        draw = self.random_stream.rand(*self.PARAMETER_RANGES.shape)
        mult_transform = (self.PARAMETER_RANGES.max(axis=-1) - self.PARAMETER_RANGES.min(axis=-1))
        mult_transform = np.expand_dims(mult_transform, axis=-1)
        add_transform = self.PARAMETER_RANGES.min(axis=-1)
        add_transform = np.expand_dims(add_transform, axis=-1)

        draw.sort(axis=-1)

        sampled_ranges = draw*mult_transform + add_transform

        assert self.check_ranges(sampled_ranges, self.PARAMETER_RANGES)

        return sampled_ranges

    def check_ranges(self, sampled, edges):
        all_good = True
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                for k in range(edges.shape[2]):
                    # lower range must be larger or equal to lower edge
                    all_good &= (sampled[i,j,k,0] >= edges[i,j,k,0])
                    # upper range must be smaller or equal to upper edge
                    all_good &= (sampled[i,j,k,1] <= edges[i,j,k,1])
                    if not all_good:
                        print('range ',edges[i,j,k])
                        print('sample',sampled[i,j,k])
                        print()

        return all_good
                




    def get_experiment(self, N):
        
        percent_A = 0.2

        percent_B = 0.2


        # States go S, P, L
        # 

        # A - 10 in A
        tA = np.array([[[0.1, 0.9, 0.0], 
                        [0.1, 0.9, 0.0]],

                        [[0, -1, -1],
                        [-1, -1, 0]],

                        [[0, 0.4, 0.6],
                        [0.0, 0.4, 0.6]]
                        ])
        # B - 10 in B
        tB = np.array([[[0.9, 0.1, 0.0], 
                        [0.9, 0.1, 0.0]],
                        [[0, -1, -1],
                        [-1, -1, 0]],
                        [[0, 0.4, 0.6],
                        [0.0, 0.4, 0.6]]
                        ])


        # C - 30 in C
        tC = np.array([[[0.1, 0.9, 0.0], 
                        [0.1, 0.9, 0.0]],
                        [[0, -1, -1],
                        [-1, -1, 0]],
                        [[0, 0.4, 0.6],
                        [0.0, 0.4, 0.6]]
                        ])

        

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


    # env has only binary actions so random is easy to generate
    def random_agent_action(self):
        actions = np.zeros(self.N)
        choices = np.random.choice(np.arange(self.N), int(self.B), replace=False)
        actions[choices] = 1
        return actions


    # a_agent should correspond to an action respresented in the transition matrix
    # a_nature should be a probability in the range specified by self.parameter_ranges
    def step(self, a_agent, a_nature):
        # print('a_nature',a_nature)
        for arm_i in range(a_nature.shape[0]):
            for arm_a in range(a_nature.shape[1]):
                param = a_nature[arm_i, arm_a]
                arm_state = int(self.current_full_state[arm_i])
                

                if param < self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0]:
                    print("Warning! nature action below allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
                    print("Setting to lower bound of range...")
                    print('arm state',arm_state)
                    param = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0]
                elif param > self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 1]:
                    print("Warning! nature action above allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
                    print("Setting to upper bound of range...")
                    print('arm state',arm_state)
                    param = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 1]
                
                # semi-annoying specific code to make sure we set the right entries for each state
                if arm_state == 0:
                    # in both action cases we'll set the p of changing to state 0
                    self.T[arm_i,arm_state,arm_a,0] = param
                    self.T[arm_i,arm_state,arm_a,1] = 1-param

                elif arm_state == 1:
                    # if action is 0 set the p of changing to state 2 (worse state)
                    if arm_a == 0:
                        self.T[arm_i,arm_state,arm_a,2] = param
                        self.T[arm_i,arm_state,arm_a,1] = 1-param

                    # if action is 1 set the p of changing to state 0 (better state)
                    elif arm_a == 1:
                        self.T[arm_i,arm_state,arm_a,0] = param
                        self.T[arm_i,arm_state,arm_a,1] = 1-param

                elif arm_state == 2:
                    # in both action cases we'll set the p of changing to state 2
                    self.T[arm_i,arm_state,arm_a,2] = param
                    self.T[arm_i,arm_state,arm_a,1] = 1-param
                else:
                    raise ValueError('Got incorrect state')



        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=int(self.current_full_state[i])
            # print(self.T[i, current_arm_state, int(a_agent[i]), :])
            # print('i',i, 'current_arm_state',current_arm_state, 'a_agent', a_agent)
            next_arm_state=np.argmax(self.random_stream.multinomial(1, self.T[i, current_arm_state, int(a_agent[i]), :]))
            next_full_state[i]=next_arm_state
            rewards[i] = self.R[i, next_arm_state]

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None

    # a_agent should correspond to an action respresented in the transition matrix
    # a_nature should be a probability in the range specified by self.parameter_ranges
    def get_T_for_a_nature(self, a_nature_expanded):

        for arm_i in range(a_nature_expanded.shape[0]):
            for arm_state in range(a_nature_expanded.shape[1]):
                for arm_a in range(a_nature_expanded.shape[2]):

                    param = a_nature_expanded[arm_i, arm_state, arm_a]

                    if param < self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0] or param > self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 1]:
                        raise ValueError("Nature setting outside allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
                        # print("Warning! nature action below allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
                        # print("Setting to lower bound of range...")
                        # param = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0]
                    # elif 
                    #     print("Warning! nature action above allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
                    #     print("Setting to upper bound of range...")
                    #     param = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 1]
                    
                    # semi-annoying specific code to make sure we set the right entries for each state
                    if arm_state == 0:
                        # in both action cases we'll set the p of changing to state 0
                        self.T[arm_i,arm_state,arm_a,0] = param
                        self.T[arm_i,arm_state,arm_a,1] = 1-param

                    elif arm_state == 1:
                        # if action is 0 set the p of changing to state 2 (worse state)
                        if arm_a == 0:
                            self.T[arm_i,arm_state,arm_a,2] = param
                            self.T[arm_i,arm_state,arm_a,1] = 1-param

                        # if action is 1 set the p of changing to state 0 (better state)
                        elif arm_a == 1:
                            self.T[arm_i,arm_state,arm_a,0] = param
                            self.T[arm_i,arm_state,arm_a,1] = 1-param

                    elif arm_state == 2:
                        # in both action cases we'll set the p of changing to state 2
                        self.T[arm_i,arm_state,arm_a,2] = param
                        self.T[arm_i,arm_state,arm_a,1] = 1-param
                    else:
                        raise ValueError('Got incorrect state')

        return np.copy(self.T)


    # this is easier to attach to environment code
    def bound_nature_actions(self, a_nature_flat, state=None, reshape=True):
        
        # num arms by num actions
        a_nature = a_nature_flat.reshape(self.N, self.T.shape[2])    

        a_nature_bounded = np.zeros(a_nature.shape)
        for arm_i in range(a_nature.shape[0]):
            for arm_a in range(a_nature.shape[1]):
                
                param = a_nature[arm_i,arm_a]

                arm_state = int(state[arm_i])
                lb = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0]
                ub = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 1]

                # print('range',lb, ub)
                # print('param in',param)
                # print('arm state',arm_state)

                a_nature_bounded[arm_i,arm_a] = ((self.tanh(torch.as_tensor(param, dtype=torch.float32))+1)/2)*(ub - lb) + lb
                # print('param out', a_nature_bounded[arm_i,arm_a])
                # print()

        if not reshape:
            a_nature_bounded = a_nature_bounded.reshape(*a_nature_flat.shape)

        return a_nature_bounded




    def reset(self):
        # self.current_full_state = np.zeros(self.N, dtype=int)
        self.current_full_state = self.random_stream.choice(self.observation_space, self.N)
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def reset_random(self):
        return self.reset()

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            # print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]










class CounterExampleRobustEnv(gym.Env):
    def __init__(self, N, B, seed):#, REWARD_BOUND):

        S = 2
        A = 2
        # N = 3
        assert N%3 == 0

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)
        self.observation_dimension = 1
        self.action_dimension = 1
        self.action_dim_nature = N
        # self.REWARD_BOUND = REWARD_BOUND
        # self.reward_range = (0, REWARD_BOUND)
        self.S = S
        self.A = A
        self.B = B
        self.init_seed = seed


        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        self.PARAMETER_RANGES = self.get_parameter_ranges(self.N)

        # make sure to set this whenever environment is created, but do it outside so it always the same
        self.sampled_parameter_ranges = None 


        self.seed(seed=seed)
        self.T, self.R, self.C = self.get_experiment(N)

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()



    # new version has one range per state, per action
    # We will sample ranges from within these to get some extra randomness
    def get_parameter_ranges(self, N):
        
        # A - 10 in A - 0, middle
        rangeA = [0, 1]

        # B - 10 in B - 1, bottom
        rangeB = [0.05, 0.9]
        # rangeB = [0.05, 0.65] # nudge the middle a bit lower so RL learns the middle policy exactly

        # C - 30 in C - 2, top
        rangeC = [0.1, 0.95]
        # rangeC = [0.35, 0.95] # nudge the middle a bit higher so RL learns the middle policy exactly


        parameter_ranges = []

        i = 0
        while i < N:
            if i%3 == 0:
                parameter_ranges.append(rangeA)
            if i%3 == 1:
                parameter_ranges.append(rangeB)
            if i%3 == 2:
                parameter_ranges.append(rangeC)
            i+=1

        # self.parameter_ranges = np.array(parameter_ranges)

        return np.array(parameter_ranges)


    def sample_parameter_ranges(self):

        return np.copy(self.PARAMETER_RANGES)

    def get_experiment(self, N):
        
        # States go S, P, L
        # 

        # A - 10 in A
        t = np.array([[ [0.5, 0.5], 
                        [0.5, 0.5]],

                       [[1.0, 0.0],
                        [0.0, -1.]] # only set the param for acting in state 1 
                     ])

        T = []
        for i in range(N):
            T.append(t)

        T = np.array(T)
        R = np.array([[0, 1] for _ in range(N)])
        C = np.array([0, 1])


        return T, R, C


    # env has only binary actions so random is easy to generate
    def random_agent_action(self):
        actions = np.zeros(self.N)
        choices = np.random.choice(np.arange(self.N), int(self.B), replace=False)
        actions[choices] = 1
        return actions


    # a_agent should correspond to an action respresented in the transition matrix
    # a_nature should be a probability in the range specified by self.parameter_ranges
    def step(self, a_agent, a_nature):

        for arm_i in range(a_nature.shape[0]):
            param = a_nature[arm_i]
            arm_state = int(self.current_full_state[arm_i])
            

            if param < self.sampled_parameter_ranges[arm_i, 0]:
                print("Warning! nature action below allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i]))
                print("Setting to lower bound of range...")
                param = self.sampled_parameter_ranges[arm_i, 0]
            elif param > self.sampled_parameter_ranges[arm_i, 1]:
                print("Warning! nature action above allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i]))
                print("Setting to upper bound of range...")
                param = self.sampled_parameter_ranges[arm_i, 1]
            
            self.T[arm_i,1,1,1] = param
            self.T[arm_i,1,1,0] = 1-param



        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=int(self.current_full_state[i])
            next_arm_state=np.argmax(self.random_stream.multinomial(1, self.T[i, current_arm_state, int(a_agent[i]), :]))
            next_full_state[i]=next_arm_state
            rewards[i] = self.R[i, next_arm_state]

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None

    # a_agent should correspond to an action respresented in the transition matrix
    # a_nature should be a probability in the range specified by self.parameter_ranges
    def get_T_for_a_nature(self, a_nature_expanded):

        for arm_i in range(a_nature_expanded.shape[0]):

            param = a_nature_expanded[arm_i]

            if param < self.sampled_parameter_ranges[arm_i, 0] or param > self.sampled_parameter_ranges[arm_i, 1]:
                raise ValueError("Nature setting outside allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i]))
                # print("Warning! nature action below allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, arm_state, arm_a]))
                # print("Setting to lower bound of range...")
                # param = self.sampled_parameter_ranges[arm_i, arm_state, arm_a, 0]

            self.T[arm_i,1,1,1] = param
            self.T[arm_i,1,1,0] = 1-param

        return np.copy(self.T)


    # this is easier to attach to environment code
    # RETUNR HERE WHEN DONE
    def bound_nature_actions(self, a_nature_flat, state=None, reshape=True):
        
        # num arms by num actions
        a_nature = a_nature_flat.reshape(self.N)    

        a_nature_bounded = np.zeros(a_nature.shape)
        for arm_i in range(a_nature.shape[0]):
            
            param = a_nature[arm_i]

            lb = self.sampled_parameter_ranges[arm_i, 0]
            ub = self.sampled_parameter_ranges[arm_i, 1]

            a_nature_bounded[arm_i] = ((self.tanh(torch.as_tensor(param, dtype=torch.float32))+1)/2)*(ub - lb) + lb

        if not reshape:
            a_nature_bounded = a_nature_bounded.reshape(*a_nature_flat.shape)

        return a_nature_bounded


    def reset_random(self):
        return self.reset()

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
            # print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]






class FakeT:
    def __init__(self, shape):
        self.shape=shape

class SISRobustEnv(gym.Env):
    def __init__(self, N, B, pop_size, seed):

        S = pop_size+1
        A = 3

        self.N = N
        self.observation_space = np.arange(S)
        self.action_space = np.arange(A)

        self.observation_dimension = 1
        self.action_dimension = 1
        self.action_dim_nature = N*4

        self.S = S
        self.A = A
        self.B = B
        self.init_seed = seed
        self.pop_size = pop_size

        self.current_full_state = np.zeros(N)
        self.random_stream = np.random.RandomState()

        self.PARAMETER_RANGES = self.get_parameter_ranges(self.N)

        # make sure to set this whenever environment is created, but do it outside so it always the same
        self.sampled_parameter_ranges = None 

        # this model only needs its params set once at the beginning
        self.param_setting = None


        self.seed(seed=seed)
        self.T, self.R, self.C = self.get_experiment(N)

        self.tanh = torch.nn.Tanh()
        self.sigmoid = torch.nn.Sigmoid()


    def p_i_s(self, q_t, i, s, pop_size):
        prob = 0
        if s == pop_size:
            if i == 0:
                prob = 1
            else:
                prob = 0
        else:
            binom = comb(s, i)
            prob = binom * q_t**(i)*(1-q_t)**(s - i)

        return prob


    def compute_distro(self, arm, s, a):

        # p(infect | contact), lower is better
        r_t = self.param_setting[arm, 0] #np.random.rand()*(r_t_range[1] - r_t_range[0]) + r_t_range[0]
        # number of contacts for delta_t, lower is better
        lam = self.param_setting[arm, 1] #np.random.rand()*(lam_range[1] - lam_range[0]) + lam_range[0]
        # action effect, larger is better
        a_effect_1 = self.param_setting[arm, 2] #np.random.rand()*(a_effect_1_range[1] - a_effect_1_range[0]) + a_effect_1_range[0]
        # action effect, larger is better
        a_effect_2 = self.param_setting[arm, 3] #np.random.rand()*(a_effect_2_range[1] - a_effect_2_range[0]) + a_effect_2_range[0]
        
        # print('r_t',r_t)
        # print('lam',lam)
        # print('a',a_effect_1)
        # print()

        delta_t = 1

        poisson_param = lam*delta_t

        S = self.S
        A = self.A
        pop_size = self.pop_size

        distro = np.zeros(S,dtype=np.float64)
        
        beta_t = (pop_size - s)/pop_size
        EPS = 1e-7
        
        q_t = None

        if a == 0:
            q_t = 1 - np.e**(-poisson_param * beta_t * r_t) 
        elif a == 1:
            q_t = 1 - np.e**(-poisson_param * beta_t/(a_effect_1) * r_t) 
        elif a == 2:
            q_t = 1 - np.e**(-poisson_param * beta_t * r_t/(a_effect_2)) 

        for sp in range(S):
            # print('s:',s)
            # print('sp:',sp)
            # print('pop_size:',pop_size)
            # print(q_t)
            # print(pop_size - s)
            # print()
            if pop_size - s <= sp and sp <= pop_size:
                # print("Here")
                # print('s:',s)
                # print('sp:',sp)
                num_infected = pop_size - sp
                prob = self.p_i_s(q_t, num_infected, s, pop_size)
                # print(prob)
                # print()
                distro[sp] = prob

        inds = distro < EPS
        distro[inds] = 0
        distro = distro / distro.sum()

        return distro


    # We will sample ranges from within these to get some extra randomness
    def get_parameter_ranges(self, N):
        
        # Wee have four params

        r_t_range = [0.5, 0.99]
        # This should scale with the number of people?
        lam_range = [1, 10] # people per day
        a_effect_1_range = [1, 10] # multiplicative effect on each parameter
        a_effect_2_range = [1, 10] # multiplicative effect on each parameter


        parameter_ranges = np.array([
            [
                r_t_range, 
                lam_range, 
                a_effect_1_range, 
                a_effect_2_range
            ] for _ in range(N)
        ])


        return parameter_ranges


    def sample_parameter_ranges(self):

        draw = self.random_stream.rand(*self.PARAMETER_RANGES.shape)
        mult_transform = (self.PARAMETER_RANGES.max(axis=-1) - self.PARAMETER_RANGES.min(axis=-1))
        mult_transform = np.expand_dims(mult_transform, axis=-1)
        add_transform = self.PARAMETER_RANGES.min(axis=-1)
        add_transform = np.expand_dims(add_transform, axis=-1)

        draw.sort(axis=-1)

        sampled_ranges = draw*mult_transform + add_transform

        assert self.check_ranges(sampled_ranges, self.PARAMETER_RANGES)

        return sampled_ranges


    def check_ranges(self, sampled, edges):
        all_good = True
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                # lower range must be larger or equal to lower edge
                all_good &= (sampled[i,j,0] >= edges[i,j,0])
                # upper range must be smaller or equal to upper edge
                all_good &= (sampled[i,j,1] <= edges[i,j,1])
                if not all_good:
                    print('range ',edges[i,j])
                    print('sample',sampled[i,j])
                    print()

        return all_good


    def get_experiment(self, N):
        
        # States go S, P, L
        # 

        # A - 10 in A
        # t = np.array([[ [0.5, 0.5], 
        #                 [0.5, 0.5]],

        #                [[1.0, 0.0],
        #                 [0.0, -1.]] # only set the param for acting in state 1 
        #              ])

        # T = []
        # for i in range(N):
        #     T.append(t)
        # T = np.array(T)

        # In general we won't want to store this T matrix...
        T = FakeT((N,self.S,self.A,self.S))
        R = np.array([np.linspace(0, 1, self.S) for _ in range(N)])
        C = np.array([0, 1, 2])


        return T, R, C


    # Fast random, inverse weighted, works for multi-action
    def random_agent_action(self):

        actions = np.zeros(self.N,dtype=int)

        current_action_cost = 0
        process_order = np.random.choice(np.arange(self.N), self.N, replace=False)
        for arm in process_order:
            
            # select an action at random
            num_valid_actions_left = len(self.C[self.C<=self.B-current_action_cost])
            p = 1/(self.C[self.C<=self.B-current_action_cost]+1)
            p = p/p.sum()
            p = None
            a = np.random.choice(np.arange(num_valid_actions_left), 1, p=p)[0]
            current_action_cost += self.C[a]
            # if the next selection takes us over budget, break
            if current_action_cost > self.B:
                break

            actions[arm] = a

        return actions



    def set_params(self, a_nature):
        
        # only set this once
        param_setting = np.zeros(self.sampled_parameter_ranges.shape[:-1])
        for arm_i in range(a_nature.shape[0]):
            for param_i in range(a_nature.shape[1]):
                param = a_nature[arm_i, param_i]

                if param < self.sampled_parameter_ranges[arm_i, param_i, 0]:
                    print("Warning! nature action below allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, param_i]))
                    print("Setting to lower bound of range...")
                    param = self.sampled_parameter_ranges[arm_i, param_i, 0]
                    raise ValueError('bad setting')
                elif param > self.sampled_parameter_ranges[arm_i, param_i, 1]:
                    print("Warning! nature action above allowed param range. Was %s but should be in %s"%(param, self.sampled_parameter_ranges[arm_i, param_i]))
                    print("Setting to upper bound of range...")
                    param = self.sampled_parameter_ranges[arm_i, param_i, 1]
                    raise ValueError('bad setting')

                param_setting[arm_i, param_i] = param
        self.param_setting = param_setting
                
                

    # a_agent should correspond to an action respresented in the transition matrix
    # a_nature should be a probability in the range specified by self.parameter_ranges
    def step(self, a_agent, a_nature):

        self.set_params(a_nature)

        ###### Get next state
        next_full_state = np.zeros(self.N, dtype=int)
        rewards = np.zeros(self.N)
        for i in range(self.N):
            current_arm_state=int(self.current_full_state[i])

            distro = self.compute_distro(i, current_arm_state, int(a_agent[i])) # self.T[i, current_arm_state, int(a_agent[i]), :]

            next_arm_state=np.argmax(self.random_stream.multinomial(1, distro))
            next_full_state[i]=next_arm_state
            rewards[i] = self.R[i, next_arm_state]

        self.current_full_state = next_full_state
        next_full_state = next_full_state.reshape(self.N, self.observation_dimension)

        return next_full_state, rewards, False, None


    # only do this if you are sure the state space is small enough (e.g., less than ~500)
    def get_T_for_a_nature(self, a_nature):

        self.set_params(a_nature)
        T = np.zeros((self.N,self.S,self.A,self.S),dtype=np.float64)
        for arm_i in range(self.N):
            for s in range(self.S):
                for a in range(self.A):
                    T[arm_i, s, a] = self.compute_distro(arm_i, s, a)

        return T


    # this is easier to attach to environment code
    def bound_nature_actions(self, a_nature_flat, state=None, reshape=True):
        
        # num arms by num actions
        a_nature = a_nature_flat.reshape((self.N, self.sampled_parameter_ranges.shape[1]))

        a_nature_bounded = np.zeros(a_nature.shape)
        for arm_i in range(a_nature.shape[0]):
            for param_i in range(a_nature.shape[1]):
                
                param = a_nature[arm_i, param_i]

                lb = self.sampled_parameter_ranges[arm_i, param_i, 0]
                ub = self.sampled_parameter_ranges[arm_i, param_i, 1]

                a_nature_bounded[arm_i, param_i] = ((self.tanh(torch.as_tensor(param, dtype=torch.float32))+1)/2)*(ub - lb) + lb

        if not reshape:
            a_nature_bounded = a_nature_bounded.reshape(*a_nature_flat.shape)

        return a_nature_bounded


    def reset_random(self):
        return self.reset()

    def reset(self):
        # self.current_full_state = np.zeros(self.N, dtype=int)
        # tested this, it's about half as fast as randint
        # self.current_full_state = self.random_stream.choice(self.observation_space, self.N)
        self.current_full_state = self.random_stream.randint(low=0, high=self.S, size=self.N)
        return self.current_full_state.reshape(self.N, self.observation_dimension)

    def render(self):
        return None

    def close(self):
        pass

    def seed(self, seed=None):
        seed1 = seed
        if seed1 is not None:
            self.random_stream.seed(seed=seed1)
            # print('seeded with',seed1)
        else:
            seed1 = np.random.randint(1e9)
            self.random_stream.seed(seed=seed1)

        return [seed1]
