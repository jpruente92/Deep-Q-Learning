### Algorithm:
 The used algorithm is a variant of Q_learning. It uses fixed targets, which means that there are two neural networks for approximating the Q-values.
 The local network is used by the agent to compute the Q-values for acting. The target network computes Q-values for the next state that are used in the learning process.
 Just the local network is trained with the TD-error computed with the SARSAmax formula. The target network can be soft updated every time the local network is trained,
 which means that the weights of the target network are set to a convex combination of the weihts of both networks with parameter TAU. The other option is that the values
 from the local network are copied to the target network all UPDATE_TARGET_EVERY steps. This is called hard update. The parameter SOFT_UPDATE controls, which of the two variants is used.
  The algorithm uses an experience replay buffer, in which
 the experience from the environment is stored. One element of experience consists of a state, an action, the resulting next state, the resulting reward and whether or not it is a
 terminal state. Every time the local network is trained, it samples a minibatch from the replay buffer with equal probability and uses it for computing TD-errors for learning.
 The parameter DOUBLE_Q controls, whether Double Q learning is used or not. If Double Q learning is used, the target value is computed by first determining the best action with
 the local network and afterwards using the value from the target network for the next state and the previously computed action. This reduces the number of learning steps, but increases the running time due to
 additional evaluations. The parameter PRIORITIZED_EXP_REPLAY controls, whether prioritized experience replay is used or not. If it is true, we save a priority value for every
 element in the replay buffer. This is used as a weight in the sampling process. The priority used here is the absolute value of the TD-error. Because sampling with weights is very time consuming for large number of elements in the
 replay buffer, we used in this case the sum tree queue data structure, which consists of a queue that stores all nodes and a sum tree. The sum tree is used for sampling
 efficiently and the queue for understanding, which node to delete, when the buffersize is reached. The idea of using sum trees comes form the paper "PRIORITIZED EXPERIENCE REPLAY"
 by Schaul et al.. To neutralize a bias that is introduced by the weighted sampling the TD-errors are multiplied with an importance sampling weight before training the network.
 The parameter B controls how much of this weight is used and B is increased over time. For new experiences, the priority is set to the maximum priority that is observed so far.
 After an experience is used, the priority is updated to the absolute value of the TD-error.


    


### Results:
The following figure displays the score of the agent during the learning process over 625 episodes, each consisting of at most 1000 steps. The average score per episode in the episodes 525-625 is 13.0 and therefore the environment is considered solved after 525 episodes. 
![alt text](./BANANA_Scores.png)



### Observations:
- Double Q Learning reduces the number of learning steps, but increases the running time
- Prioritized experience replay increases the running time but is not always beneficial e.g. for LunarLander it does not perform well. For the Banana environment it educes the number of learning steps 


### Future Improvements:
- include ideas from the paper "NOISY NETWORKS FOR EXPLORATION" by Fortunato et al.
- include other ideas from the paper "Rainbow: Combining Improvements in Deep Reinforcement Learning" by Hessel et al.
    
    

