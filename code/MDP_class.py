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
            if np.sum(self.__f_trans[cur_state, action, :i]) <= eps < np.sum(self.__f_trans[cur_state, action, :i + 1]):
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
    
    def get_max_action_value(self, values):
        Q_pi = np.zeros((self.nstates, self.nactions))
        for s in range(self.nstates):
            for a in range(self.nactions):
                Q_pi[s, a] = np.sum(self.__f_trans[s, a, :] * (self.__f_reward[s, a, :] +
                                                            self.gamma * values))
        # Return the maximizers of the action value function over the actions a
        Q_max = np.zeros_like(values, dtype=int)
        for s in range(self.nstates):
            # Action that maximizes Q for given pi
            Q_max[s] = np.argmax(Q_pi[s, :])
        return Q_max

    def evaluate_policy(self, pi):
        # Assuming a single terminal state S-1, remove it from
        # policy evaluation to get full rank A matrix
        # Init coeffcient matrix based on diagonal elements having + 1 term
        states = self.nstates
        A = np.identity(states)
        # Assign as per bellman's policy eval equations
        for s in range(states):
            A[s, :] = A[s, :] - self.gamma * self.__f_trans[s, pi[s], :states]
        # Assign right side b vector as sum of T * R terms
        b = np.zeros(states)
        for s in range(states):
            b[s] = np.sum(self.__f_trans[s, pi[s], :states] * self.__f_reward[s, pi[s], :states])
        # Check if it is an episodic task, in which case we already know
        # value for terminal state = 0 (enforce it)
        if self.type == 'episodic':
            A = A[:-1, :-1]
            b = b[:-1]
        # Solve and return Ax = b
        values = np.linalg.solve(A, b)
        if self.type == 'episodic':
            # For last state s = |S| - 1
            values = np.append(values, 0)
        return values

    # Perform MDP planning using Howard's policy iteration algo
    # Have assumed that last state is unique terminal state
    def plan(self):
        # Initialise a random prev and current policy vector
        pi_prev = np.random.randint(0, self.nactions, self.nstates)
        pi_cur = np.copy(pi_prev)
        # Change 1 action in pi_cur to enter loop (assuming at least 2 actions in MDP)
        if pi_cur[0] != 0:
            pi_cur[0] = 0
        else:
            pi_cur[0] = 1
        # Init values array
        values = np.zeros_like(pi_prev, dtype=float)
        # Begin policy iteration/improvement loop
        while not np.array_equal(pi_prev, pi_cur):
            # Update pi_prev to pi_cur
            pi_prev = pi_cur
            # Get current performance
            # If episodic V(|S|-1) == 0 fixed and solver solves accordingly
            values = self.evaluate_policy(pi_prev)
            # Attempt to improve policy by evaluating action value functions
            pi_cur = self.get_max_action_value(values)
        return values, pi_cur
