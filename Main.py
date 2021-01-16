from Environment import environment
from QLearning import *

# modifying the environment
problem_name = "Banana.exe"
type = "unity"

# problem_name = 'LunarLander-v2'
# type = "gym"
seed = 0
env = environment(type, problem_name, seed)

# starting the agent
start_agent(env, seed)
