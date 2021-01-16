# here all hyperparameters can be set globally

# output and loading specifications
LOAD = True  # loading neural networks from file
# possible filenames:   1 Banana: Double Q
#                       2 Banana_pexpr: Double Q + prioritized experience replay
#                       3 Banana_duel: Double Q
#                       4 Banana_duel_pexpr: Double Q + prioritized experience replay + dualing neural network
FILENAME_FOR_LOADING = "Banana_pexpr"  # set name for loading files for model weights
SAVE = False  # save neural networks to file
FILENAME_FOR_SAVING = ""  # set name for saving files for model weights
PLOT = False  # plot the scores
PLOTNAME = "Lunar"  # name for the plot file
WITH_PROFILING = False  # print running times of different tasks of the algorithm
ENV_TRAIN = False  # train mode for environment
VAL_ENV_SOLVED = 13.0  # stop when average over 100 episodes is this value, banana env is considered solved with 13.0

# which version of algortihm to apply
SOFT_UPDATE = True  # if true updates q targets softly, otherwise every hard with UPDATE_TARGET_EVERY
DOUBLE_Q = True  # applying double q learning
PRIORITIZED_EXP_REPLAY = True  # applying prioritized experience replay
DUELING_NETWORK = False  # applying dueling network

# hyperparameters of the algorithm
NR_EPISODES = 2000
MAX_NR_STEPS = 1000
EPS_START = 0.01
EPS_END = 0.01
EPS_DECAY = 0.995
B_START = 0.6
A = 0.6  # parameter for prioritized experience replay
BUFFER_SIZE = 10000  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 0.001  # for soft update of target parameters
LR = 0.0005  # learning rate
UPDATE_EVERY = 4  # how often to update the network
UPDATE_TARGET_EVERY = 20  # how often to do the hard update for targets

# network architecture
hidden_layers = [64, 64]  # hidden layers for the non dueling architecture
