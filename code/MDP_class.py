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
        # Collection of states as numpy array for quick sampling
        self.states = np.arange(self.nstates)
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
            # Error encountered in parsing line, message printed
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
            # Error in line, message printed
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
        elif self.gamma > 1.0 or self.gamma < 0.0:
            print("Discount factor gamma out of range on line {0}".format(fin.line - 1))
            exit(-1)
        elif self.gamma == 1.0 and self.type == 'continuing':
            print("Discount factor cannot be 1.0 for continuing MDP: line {0}".format(fin.line - 1))
            exit(-1)
        # List to keep track of all terminal states in an episodic MDP
        # terminal states in a continuing MDP don't need to be handled specially
        self.terminal = []
        self.non_terminal = list(range(self.nstates))
        # Return all terminal state candidates as per format,
        # for an episodic task, there must be at least one terminal state, non-empty in case of an episodic task
        # Transitions to itself with probability 1 for all actions implies terminal state
        if self.type == 'episodic':
            for s in range(self.nstates):
                if np.array_equal(self.__f_trans[s, :, s], np.ones(self.nactions)):
                    # Add to list of terminal states and remove from non terminal
                    self.terminal.append(s)
                    self.non_terminal.remove(s)
            # Check if episodic and collection of terminal states empty
            if not self.terminal:
                print("Error in MDP definition, episodic tasks must have at least one terminal state")
                exit(-1)
        # Record largest reward values Rmax, accessible to algorithms
        self.Rmax = np.max(self.__f_reward)

    # Function to query MDP as an available sample model,
    # The MDP object also keeps track of agents state in environment
    def sample(self, cur_state, action):
        # Get next state as per transition function distribution
        next_state = np.random.choice(self.states, p=self.__f_trans[cur_state, action, :])
        # Get reward associated with transition
        rew = self.__f_reward[cur_state, action, next_state]
        # Check if Episode has ended
        epi_ended = False
        # self.terminal is an empty list for continuing tasks
        if next_state in self.terminal:
            epi_ended = True
        # Return reward, new current state and end status to algorithm
        return rew, next_state, epi_ended
    
    # Does not consider terminal and non-terminal states separately
    def get_greedy_policy(self, values):
        # Compute the action value function for pi
        Q_pi = np.sum(self.__f_trans * (self.__f_reward + self.gamma * values), axis=2)
        # Return the greedy policy wrt current action values
        pi_greedy = np.argmax(Q_pi, axis=1)
        return pi_greedy

    # Evaluate policy on true MDP or current model
    def evaluate_policy(self, pi):
        # The value associated with terminal states is constrained to be = 0.0 in case of episodic tasks
        # coefficient matrix based on bellman's policy eval equations
        states = self.nstates
        A = np.identity(states) - self.gamma * self.__f_trans[self.states, pi, :]
        # Assign right side b vector as sum of T * R terms
        b = np.sum(self.__f_trans[self.states, pi, :] * self.__f_reward[self.states, pi, :], axis=1)
        values = np.zeros(states, dtype=np.float32)
        # Get list of non terminal states
        no_terminal_state_lst = list(range(states))
        # Check if it is an episodic task, in which case we already know
        # value for terminal states are = 0 (enforce it) (only enter if list non empty)
        if self.terminal:
            # New delete rows and columns corresponding to indices in self.terminal
            A = np.delete(np.delete(A, self.terminal, axis=1), self.terminal, axis=0)
            b = np.delete(b, self.terminal, axis=0)
        # Solve and return Ax = b for non_terminal states, fixed to 0.0 for terminal states
        values[self.non_terminal] = np.linalg.solve(A, b)
        return values

    # Perform MDP planning using Howard's policy iteration algorithm
    # Optimal policy for terminal states is arbitrary in some sense
    # However due to nature of implementation, the optimal action is the one
    # with the largest 1 step reward
    def plan(self):
        # Initialise a random prev and current policy vector
        pi_prev = np.random.randint(0, self.nactions, self.nstates)
        # Perform 1 iteration of HPI to enter while loop
        # Get value function for current policy
        values = self.evaluate_policy(pi_prev)
        # Attempt to improve policy by evaluating action value functions
        pi_cur = self.get_greedy_policy(values)
        # Begin policy iteration/improvement loop
        while not np.array_equal(pi_prev, pi_cur):
            # Update pi_prev to pi_cur
            pi_prev = pi_cur
            # Get value function for current policy
            values = self.evaluate_policy(pi_prev)
            # Attempt to improve policy by evaluating action value functions
            pi_cur = self.get_greedy_policy(values)
        return values, pi_cur
