# Deep Q-Learning

This project is an implementation of a Q-learning algorithm for solving unity or gym environments.

### Required packages:
- (gym if solving gym environment)
- numpy
- python (version 3.6)
- pytorch
- unityagents if solving unity environment

### Required files:
The unity exe file has to be inside this folder; here is an environment called "Bananas.exe" included.

### Bananas environment:
The included Bananas environement is an environment, in which the agent moves inside a square. 
Inside the square, there are yellow bananas giving a reward of 1 and blue bananas giving a reward of -1. The state space consists of 37 dimensions and 4 actions can be taken (move forward, move backwards, turn left, turn right). The environment is considered solved, when an average score of 13.0 over 100 episodes is reached.

### Starting the program:
- hyperparameters and settings for the algorithm can be changed in the file "Hyperparameter.py".
    -> for viewing a trained agent set "LOAD" to True,"FILENAME_FOR_LOADING" to the name of the files of the model
    weights(without "_model_local.pth" or "_model_target.pth"),  "EPS_START" to 0.01 and for unity "ENV_TRAIN" to False.
    -> for training a new agent set "LOAD" to False, "EPS_START" to 1.00 and for unity "ENV_TRAIN" to True and if you want to
    save the model weights, set "Save" to True and "FILENAME_FOR_SAVING" to the name for the weight files
- use the file "Main.py".
    -> comment out the method you want to start (either gym or unity).
    -> run the file "Main.py".
- other gym or unity environments can be used as well. Note that you cannot load a trained version for new environments until one is saved.
- changes in all other files are not recommended.


