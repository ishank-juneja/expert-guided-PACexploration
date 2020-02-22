import numpy as np
from random import randint, random


# Class for an arbitrary finite MDP instance
# Reads in S,A,R,T,gamma from a text file
class MDP:
    # Constructor that reads a text file to
    # parse the MDP instance
    def __init__(self, file_name):
        # Open MDP instance file
        fin = open(file_name, 'r')
        # Read in number of states from line 1 as an int
        self.nstates = int(fin.readline())
        # Read in number of action types from line 2
        self.nactions = int(fin.readline())
        # Read in default born state
        self.born = int(fin.readline())
        # Read in reward function into a matrix
        # Init reward matrix, private data member, algos can't access
        self.__f_reward = np.zeros((self.nstates, self.nactions, self.nstates))
        # Read nstates x nactions number of lines
        for i in range(self.nstates * self.nactions):
            s = i // self.nactions
            a = i % self.nactions
            self.__f_reward[s][a] = \
                np.fromstring(fin.readline(), dtype=float, sep='\t')
        # Read in Transition function into a matrix, private data member, algos can't access
        self.__f_trans = np.zeros_like(self.__f_reward)
        for i in range(self.nstates * self.nactions):
            s = i // self.nactions
            a = i % self.nactions
            self.__f_trans[s][a] = \
                np.fromstring(fin.readline(), dtype=float, sep='\t')
        # Read discount factor
        self.gamma = float(fin.readline())
        # Read Problem type --> continuing or episodic
        self.type = fin.readline()[:-1]
        if self.type != 'episodic' and self.type != 'continuing':
            print("Error, provided type {0} must be in [episodic, continuing]".format(self.type))
        # List to keep track of terminal states
        self.terminal = []
        # Return all terminal state candidates As per format,
        # for an episodic task, the last state must always be a terminal state, non-empty in case of an episodic task
        # Transitions to itself with probability 1 irrespective of the action chosen by the policy imply terminal state
        for s in range(self.nstates):
            if np.array_equal(self.__f_trans[s, :, s], np.ones(self.nactions)):
                self.terminal.append(s)

    # Function to query MDP as an available sample model,
    # The MDP object also keeps track of agents state in environment
    def sample(self, cur_state, action):
        # Get next state from transition function
        # Generate random number between 0 and 1
        eps = random()
        next_state = 0
        for i in range(self.nstates):
            # If random eps lies in interval associated with ith state
            if np.sum(self.__f_trans[cur_state, action, :i]) < np.sum(self.__f_trans[cur_state, action, :i + 1]):
                next_state = i
                break
        # Get reward associated with transition
        rew = self.__f_reward[cur_state, action, next_state]
        # Episode ended
        epi_ended = False
        if next_state in self.terminal:
            epi_ended = True
        # Return reward and new state to main
        return rew, next_state, epi_ended
