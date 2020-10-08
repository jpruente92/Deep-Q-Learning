# here all hyperparameters can be set globally

# output and loading specifications
LOAD = False            # loading neural networks from file
FILENAME_FOR_LOADING="Banana"  # set name for loading files for model weights
SAVE = False             # save neural networks to file
FILENAME_FOR_SAVING="Banana2"  # set name for saving files for model weights
PLOT = False           # plot the scores
PLOTNAME = "BANANA_Scores2.png" # name for the plot file
WITH_PROFILING = False  # print running times of different tasks of the algorithm
ENV_TRAIN = True       # train mode for environment
VAL_ENV_SOLVED = 13.0    # stop when average over 100 episodes is this value



# which version of algortihm to apply
SOFT_UPDATE = True      # if true updates q targets softly, otherwise every hard with UPDATE_TARGET_EVERY
DOUBLE_Q= True          # applying double q learning
PRIORITIZED_EXP_REPLAY= False          # applying prioritized experience replay

# hyperparameters of the algorithm
NR_EPISODES=2000
MAX_NR_STEPS=1000
EPS_START=1.00
EPS_END=0.01
EPS_DECAY=0.995
B_START= 0.4
B_END=1.00
B_INCREASE=1.002
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
        self.LOAD=LOAD
        self.FILENAME_FOR_LOADING=FILENAME_FOR_LOADING
        self.SAVE=SAVE
        self.FILENAME_FOR_SAVING=FILENAME_FOR_SAVING
        self.PLOT = PLOT
        self.PLOTNAME=PLOTNAME
        self.WITH_PROFILING = WITH_PROFILING
        self.ENV_TRAIN = ENV_TRAIN
        self.VAL_ENV_SOLVED =VAL_ENV_SOLVED
        self.SOFT_UPDATE=SOFT_UPDATE
        self.UPDATE_TARGET_EVERY=UPDATE_TARGET_EVERY
        self.DOUBLE_Q=DOUBLE_Q
        self.PRIORITIZED_EXP_REPLAY=PRIORITIZED_EXP_REPLAY
        self.NR_EPISODES=NR_EPISODES
        self.MAX_NR_STEPS=MAX_NR_STEPS
        self.EPS_START=EPS_START
        self.EPS_END=EPS_END
        self.EPS_DECAY=EPS_DECAY
        self.B_START=B_START
        self.B_END=B_END
        self.B_INCREASE=B_INCREASE
        self.A=A
        self.BUFFER_SIZE=BUFFER_SIZE
        self.BATCH_SIZE=BATCH_SIZE
        self.GAMMA=GAMMA
        self.TAU=TAU
        self.LR =LR
        self.UPDATE_EVERY=UPDATE_EVERY
        self.hidden_layers=hidden_layers

