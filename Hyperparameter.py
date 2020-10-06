# here all hyperparameters can be set globally

# output and loading specifications
LOAD = True            # loading neural networks from file
SAVE = False             # save neural networks to file
PLOT = False           # plot the scores
WITH_PROFILING = False  # print running times of different tasks of the algorithm
ENV_TRAIN = False       # train mode for environment
VAL_ENV_SOLVED = 20.0    # stop when average over 100 episodes is this value



# which version of algortihm to apply
SOFT_UPDATE = True      # if true updates q targets softly, otherwise every hard with UPDATE_TARGET_EVERY
DOUBLE_Q= True          # applying double q learning
PRIORITIZED_EXP_REPLAY= False          # applying prioritized experience replay

# hyperparameters of the algorithm
NR_EPISODES=2
MAX_NR_STEPS=1
EPS_START=1.00
EPS_END=0.01
EPS_DECAY=0.995
B_START=0.4
B_END=1.00
B_INCREASE=1.005
A=0.6                   # parameter for prioritized experience replay
BUFFER_SIZE = 10000    # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001             # for soft update of target parameters
LR = 0.0005             # learning rate
UPDATE_EVERY = 4        # how often to update the network
UPDATE_TARGET_EVERY = 20 # how often to do the hard update for targets

# network architecture
hidden_layers=[64,64]

class Hyperparameters():
    def __init__(self):
        self.LOAD=LOAD  #0
        self.SAVE=SAVE  #1
        self.PLOT = PLOT    #2
        self.WITH_PROFILING = WITH_PROFILING    #3
        self.ENV_TRAIN = ENV_TRAIN  #4
        self.VAL_ENV_SOLVED =VAL_ENV_SOLVED #5
        self.SOFT_UPDATE=SOFT_UPDATE    #6
        self.UPDATE_TARGET_EVERY=UPDATE_TARGET_EVERY    #7
        self.DOUBLE_Q=DOUBLE_Q  #8
        self.PRIORITIZED_EXP_REPLAY=PRIORITIZED_EXP_REPLAY  #9
        self.NR_EPISODES=NR_EPISODES    #10
        self.MAX_NR_STEPS=MAX_NR_STEPS  #11
        self.EPS_START=EPS_START    #12
        self.EPS_END=EPS_END    #13
        self.EPS_DECAY=EPS_DECAY    #14
        self.B_START=B_START    #15
        self.B_END=B_END    #16
        self.B_INCREASE=B_INCREASE  #17
        self.A=A    #18
        self.BUFFER_SIZE=BUFFER_SIZE    #19
        self.BATCH_SIZE=BATCH_SIZE  #20
        self.GAMMA=GAMMA    #21
        self.TAU=TAU    #22
        self.LR =LR #23
        self.UPDATE_EVERY=UPDATE_EVERY  #24
        self.hidden_layers=hidden_layers    #25
        # list for getting parameters by number
        self.parameterlist=[self.LOAD,self.SAVE,self.PLOT,self.WITH_PROFILING,self.ENV_TRAIN,self.VAL_ENV_SOLVED,self.SOFT_UPDATE,
                            self.UPDATE_TARGET_EVERY,self.DOUBLE_Q,self.PRIORITIZED_EXP_REPLAY,self.NR_EPISODES,self.MAX_NR_STEPS,
                            self.EPS_START,self.EPS_END,self.EPS_DECAY,self.B_START,self.B_END,self.B_INCREASE,self.A,self.BUFFER_SIZE,
                            self.BATCH_SIZE,self.GAMMA,self.TAU,self.LR,self.UPDATE_EVERY,self.hidden_layers]
        # list for getting names of parameters by number
        self.namelist=["LOAD","SAVE","PLOT","WITH_PROFILING","ENV_TRAIN","VAL_ENV_SOLVED","SOFT_UPDATE",
                            "UPDATE_TARGET_EVERY","DOUBLE_Q","PRIORITIZED_EXP_REPLAY","NR_EPISODES","MAX_NR_STEPS",
                            "EPS_START","EPS_END","EPS_DECAY","B_START","B_END","B_INCREASE","A","BUFFER_SIZE",
                            "BATCH_SIZE","GAMMA","TAU","LR","UPDATE_EVERY","hidden_layers"]
