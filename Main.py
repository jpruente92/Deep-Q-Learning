from Hyperparameter import Hyperparameters
from Profile import Profile
from QLearning import start_agent_unity, start_agent_gym


# unity env
problem_name_unity = "Banana.exe"

# starting the agent
start_agent_unity(problem_name_unity,Hyperparameters(),seed=0)


# gym env
problem_name_gym='LunarLander-v2'

# starting the agent
# start_agent_gym(problem_name_gym,Hyperparameters(),seed=0)






