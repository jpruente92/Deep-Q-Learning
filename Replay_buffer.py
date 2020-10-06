import random
import time
from collections import deque, namedtuple

from Hyperparameter import Hyperparameters
from Sum_tree import Sum_tree_queue


class ReplayBuffer:

    def __init__(self, number_actions, seed, profile,param):
        self.param = param
        self.number_actions = number_actions
        self.PRIORITIZED_EXP_REPLAY=self.param.PRIORITIZED_EXP_REPLAY
        if self.param.PRIORITIZED_EXP_REPLAY:
            self.memory = Sum_tree_queue(self.param.BUFFER_SIZE)
        else:
            self.memory=deque(maxlen=self.param.BUFFER_SIZE)
        self.experience=namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.batch_size = self.param.BATCH_SIZE
        self.seed = random.seed(seed)
        self.profile = profile
        self.cnt=0

    def add(self, state, action, reward, next_state, done,priority,A):
        new_experience = self.experience(state, action, reward, next_state, done)
        if self.PRIORITIZED_EXP_REPLAY:
            self.memory.add_new_value(new_experience,priority**A,self.cnt)
            self.cnt=(self.cnt+1)%1000000
        else:
            self.memory.append(new_experience)

    def sample(self):
        start_time=time.time()
        self.profile.total_number_sampling_calls+=1
        if self.PRIORITIZED_EXP_REPLAY:
            samples,probabilities_of_samples=self.memory.sum_tree.sample_values(False,self.batch_size)
        else:
            samples = random.sample(self.memory, k=self.batch_size)
            probabilities_of_samples=[]

        self.profile.total_time_sampling+=(time.time() - start_time)
        return samples, probabilities_of_samples

    def __len__(self):
        return len(self.memory)
