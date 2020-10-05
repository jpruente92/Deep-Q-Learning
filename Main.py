from QLearning import start_agent_unity, start_agent_gym

# unity env
problem_name_unity = "Banana"

# start agent without profiling
# start_agent_unity(problem_name_unity+"_DOUBLE_Q_",False,False,True)

# start agent with profiling from saved version
start_agent_unity(problem_name_unity+"_DOUBLE_Q_",False,True,True)

# gym env
# problem name
# problem_name_gym='LunarLander-v2'

# start agent without profiling
# start_agent_gym(problem_name_gym+"_DOUBLE_Q_",False,False,True)

# start agent with profiling from saved version
# start_agent_gym(problem_name_gym+"_DOUBLE_Q_",False,True,True)




