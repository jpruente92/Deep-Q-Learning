# wrapper to use gym and unity environments in the same algorithms

from Hyperparameter import *


class environment():
    def __init__(self, type, problem_name, seed = 0):
        self.type = type
        if type == "unity":
            from unityagents import UnityEnvironment
            env_name = "./" + problem_name
            self.env = UnityEnvironment(file_name=env_name)
            # get the default brain
            self.brain_name = self.env.brain_names[0]
            self.brain = self.env.brains[self.brain_name]
        if type == "gym":
            import gym
            self.env = gym.make(problem_name)
            self.env.seed(seed)

    def reset(self):
        if self.type == "unity":
            env_info = self.env.reset(train_mode=ENV_TRAIN)[self.brain_name]  # reset the environment
            return env_info.vector_observations[0]
        if self.type == "gym":
            return self. env.reset()

    def step(self, action):
        if self.type == "unity":
            env_info = self.env.step(action.astype(int))[self.brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # see if episode has finished
            return next_state, reward, done, None
        if self.type == "gym":
            self.env.step(action)

    def close(self):
        self.env.close()

    def get_state_dim(self):
        if self.type == "unity":
            env_info = self.env.reset(train_mode=ENV_TRAIN)[self.brain_name]
            return len(env_info.vector_observations[0])
        if self.type == "gym":
            return self.env.observation_space.shape

    def get_nr_actions(self):
        if self.type == "unity":
            env_info = self.env.reset(train_mode=ENV_TRAIN)[self.brain_name]
            return self.brain.vector_action_space_size
        if self.type == "gym":
            return self.env.action_space.n