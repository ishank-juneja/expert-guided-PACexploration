import numpy as np
import Utils


# Class for an arbitrary finite MDP instance
# Reads in S,A,R,T,gamma and born state from an instance text file
class MDP:
    # Constructor: reads a text file to parse MDP instance
    def __init__(self, file_name):
        # Open MDP instance file
        fin = Utils.FileLineWrapper(open(file_name, 'r'))
        # Read in number of states from line 1 as an int
        self.nstates = int(fin.readline())
        # Read in number of action types from line 2
        self.nactions = int(fin.readline())
        # Read in default born state
        self.born = int(fin.readline())
        # Check if born state lies in range or is -1
        if self.born < -1 or self.born > self.nstates - 1:
            # Throw error
            print("Error in born state on line {0}".format(fin.line))
            exit(-1)
        # Read in reward function into a matrix
        # Init reward matrix, private data member, algos can't access
        self.__f_reward = np.zeros((self.nstates, self.nactions, self.nstates))
        # Read nstates x nactions number of lines
        for i in range(self.nstates * self.nactions):
            s = i // self.nactions
            a = i % self.nactions
            line_data = fin.readline()
            if Utils.check_reward(line_data, fin):
                self.__f_reward[s][a] = np.fromstring(line_data, dtype=float, sep='\t')
            # Error in line, message printed
            else:
                exit(-1)
        # Read in Transition function into a matrix, private data member, algos can't access
        self.__f_trans = np.zeros_like(self.__f_reward)
        for i in range(self.nstates * self.nactions):
            s = i // self.nactions
            a = i % self.nactions
            line_data = fin.readline()
            if Utils.check_transition(line_data, fin):
                self.__f_trans[s][a] = np.fromstring(line_data, dtype=float, sep='\t')
            # Error in line, meesage printed
            else:
                exit(-1)
        # Read discount factor
        self.gamma = float(fin.readline())
        # Read in Problem type as string --> continuing or episodic
        self.type = fin.readline()[:-1]  # exclude terminal '\n'
        # Check validity
        if self.type != 'episodic' and self.type != 'continuing':
            print("Error on line {0}, provided type {1} must be in [episodic, continuing]".format(fin.line, self.type))
            exit(-1)
        if self.gamma > 1.0 or self.gamma < 0.0:
            print("Discount factor gamma out of range on line {0}".format(fin.line - 1))
            exit(-1)
        elif self.gamma == 1.0 and self.type == 'continuing':
            print("Discount factor cannot be 1.0 for continuing MDP: line {0}".format(fin.line - 1))
            exit(-1)
        # List to keep track of all terminal states in the MDP
        self.terminal = []
        # Return all terminal state candidates as per format,
        # for an episodic task, the last state must always be a terminal state, non-empty in case of an episodic task
        # Transitions to itself with probability 1 for all actions implies terminal state
        for s in range(self.nstates):
            if np.array_equal(self.__f_trans[s, :, s], np.ones(self.nactions)):
                self.terminal.append(s)
        # Check if episodic and collection of terminal states empty
        if self.type == 'episodic' and not self.terminal:
            print("Error in MDP definition, episodic tasks must have at least one terminal state")
            exit(-1)
        # Record largest reward values Rmax, accessible to algorithms
        self.rmax = np.max(self.__f_reward)

    # Function to query MDP as an available sample model,
    # The MDP object also keeps track of agents state in environment
    def sample(self, cur_state, action):
        # Get next state as per transition function distribution
        next_states = np.arange(self.nstates)
        next_state = np.random.choice(next_states, p=self.__f_trans[cur_state, action, :])
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
                Q_pi[s, a] = np.sum(self.__f_trans[s, a, :] * (self.__f_reward[s, a, :] + self.gamma * values))
        # Return the greedy policy wrt current action values
        pi_greedy = np.zeros_like(values, dtype=int)
        for s in range(self.nstates):
            # Action that maximizes Q for given pi
            pi_greedy[s] = np.argmax(Q_pi[s, :])
        return pi_greedy

    def evaluate_policy(self, pi):
        # Assuming a single terminal state S-1, remove it from
        # policy evaluation to get full rank A matrix
        # Init coefficient matrix based on diagonal elements having + 1 term
        states = self.nstates
        A = np.identity(states)
        # Assign as per bellman's policy eval equations
        for s in range(states):
            A[s, :] = A[s, :] - self.gamma * self.__f_trans[s, pi[s], :]
        # Assign right side b vector as sum of T * R terms
        b = np.zeros(states)
        for s in range(states):
            b[s] = np.sum(self.__f_trans[s, pi[s], :] * self.__f_reward[s, pi[s], :])
        # Array to hold values
        values = np.zeros(states)
        # Get list of non terminal states
        state_lst = list(range(states))
        # Check if it is an episodic task, in which case we already know
        # value for terminal state = 0 (enforce it)
        if self.type == 'episodic':
            # New delete rows and columns corresponding to indices in self.terminal
            A = np.delete(np.delete(A, self.terminal, axis=1), self.terminal, axis=0)
            b = np.delete(b, self.terminal, axis=0)
            for i in range(states):
                if i in self.terminal:
                    state_lst.remove(i)
        # Solve and return Ax = b
        values[state_lst] = np.linalg.solve(A, b)
        return values

    # Perform MDP planning using Howard's policy iteration algorithm
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
