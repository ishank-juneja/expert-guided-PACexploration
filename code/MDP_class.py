import numpy as np


# Class for an arbitrary finite MDP instance
# Reads in S,A,R,T,gamma and born state as initialisation parameters
class MDP:
    def __init__(self, **kwargs):
        # Number of states in MDP
        self.nstates = kwargs['nstates']
        # Collection of states as numpy array for quick sampling
        self.states = np.arange(self.nstates)
        # Read in number of action types from line 2
        self.nactions = kwargs['nactions']
        # Read in default born state
        self.born = kwargs['born']
        # Read in reward function as matrix
        # Init reward matrix, private data member, algos can't access
        self.__f_reward = kwargs['rewards']
        # Transition function matrix, private data member, algos can't access
        self.__f_trans = kwargs['transitions']
        # Read discount factor
        self.gamma = kwargs['gamma']
        # Problem type is a string in [continuing or episodic]
        self.type = kwargs['problem_type']
        # Check validity
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
        self.rmax = np.max(self.__f_reward)

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
            # Switch all improvable states as per rules of HPI
            pi_cur = self.get_greedy_policy(values)
        return values, pi_cur

    # Function to perform policy improvement using simple randomized policy iteration
    # simple + random --> switch one improvable state at a time for fixed number of iterations
    # Flaw --> Possible that no update made in certain iterations due to floating point comparison
    def simpleRPI(self, pi, niters):
        # Policy that is improved wrt current mdp (or de-proved wrt mirror mdp) is returned
        # avoiding muting object (array) that was passed to function
        policy = pi.copy()
        for i in range(niters):
            # Get current values and look for improvable states
            values = self.evaluate_policy(policy)
            # Get action value function associated with current value function
            Q_pi = np.sum(self.__f_trans * (self.__f_reward + self.gamma * values), axis=2)
            # Get all improvable state action pairs under the current MDP (mirrored for de-improvement)
            # Floating point comparison may lead to current s, pi(s) being included in improvable_pairs
            improvable_pairs = np.argwhere(Q_pi > values.reshape(-1, 1))
            # Get number of improvable pairs
            npairs = improvable_pairs.shape[0]
            # Only attempt to improve if there are any improvable states
            if npairs > 0:
                # Choose a random state-action pair for improvement
                improve_pair = np.random.randint(npairs)
                # Update policy
                policy[improvable_pairs[improve_pair, 0]] = improvable_pairs[improve_pair, 1]
            else:
                # Break for loop and return policy, no longer improvable
                break
        return policy







