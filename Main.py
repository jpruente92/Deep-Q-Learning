from Hyperparameter import Hyperparameters
from Profile import Profile
from QLearning import start_agent_unity, start_agent_gym, test_parameters

# unity env
problem_name_unity = "Banana"

# starting the agent
# start_agent_unity(problem_name_unity,Hyperparameters())

# testing parameters and save to file
test_parameters("test.txt",problem_name_unity,True,Hyperparameters(),Profile(),[8,9,25],[8,9,25],[[True,False],[True,False],[[64,64],[64,64,64]]])

# gym env
problem_name_gym='LunarLander-v2'
#start_agent_gym(problem_name_gym,Hyperparameters())






