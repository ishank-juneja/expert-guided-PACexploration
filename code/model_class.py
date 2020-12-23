import numpy as np


# Model to store current understanding of an environment
# different from the MDp sample model
class Model:
    # Init model using underlying MDP sample model
    def __init__(self, mdp):
        # copy the same state and action space as actual MDP
        self.nstates = mdp.nstates
        # Array for MDP states
        self.states = mdp.states
        self.nactions = mdp.nactions
        # Init containers for eventually learned MDP parameters,
        # we learn the parameters to get a "PAC-MDP" model under the PAC-RL framework
        self.T_hat = np.zeros((mdp.nstates, mdp.nactions, mdp.nstates), dtype=np.float32)
        self.R_hat = np.zeros((mdp.nstates, mdp.nactions, mdp.nstates), dtype=np.float32)
        # Whether model is ready for deployment
        self.model_valid = False
        # Total visits to a state-action pair
        self.total_visits = np.zeros((mdp.nstates, mdp.nactions), dtype=int)
        # Total transitions made of type s-->a-->s'
        self.total_trans = np.zeros((mdp.nstates, mdp.nactions, mdp.nstates), dtype=int)
        # Total Reward obtained in a transition s,a,s'
        self.total_rew = np.zeros((mdp.nstates, mdp.nactions, mdp.nstates), dtype=np.float32)
        # Discount factor read in original MDP
        self.gamma = mdp.gamma
        # MDP type
        self.type = mdp.type
        # List of terminal states, to be excluded from model learning process
        # Both empty in case of continuing task
        self.exclude = mdp.terminal
        # List of non-terminal states
        self.non_terminal = mdp.non_terminal
        # Assign probability 1.0 of coming back to terminal state
        # under any action taken while at terminal state
        for state in self.exclude:
            # Other probabilities = 0.0 by default
            self.T_hat[state, :, state] = 1.0
        # Number of updates/time steps taken
        self.time = 0
        # Get born state from MDP sample model object
        self.born = mdp.born
        # Get current state
        self.cur_state = self.get_born_state()

    def get_born_state(self):
        # Use default born state s_o if the task specifies it
        # Else pick a random state to start with
        if -1 < self.born < self.nstates:
            return self.born
        else:
            # If -1/any impossible index passed, upper limit not included
            return np.random.randint(0, self.nstates)

    # Update model estimate
    def update(self, a, r, s_prime):
        s = self.cur_state
        # Update visitation stats
        self.total_trans[s, a, s_prime] += 1
        self.total_rew[s, a, s_prime] += r
        self.total_visits[s, a] += 1
        # Update number of steps taken
        self.time += 1
        # Update transition probabilities
        self.T_hat[s, a, :] = self.total_trans[s, a, :]/self.total_visits[s, a]
        # Update reward function
        self.R_hat[s, a, s_prime] = self.total_rew[s, a, s_prime]/self.total_trans[s, a, s_prime]
        # Model becomes valid once all non terminal state-action pairs have been reached once
        if not self.model_valid:
            # Can't take any actions from terminal state, so exclude from checking process
            # delete all rows corresponding to [s, a] where s \in terminal/excluded set
            if np.all(np.delete(self.total_visits, self.exclude, axis=0) > 0) and self.gamma < 1.0:
                print("Model became valid")
                self.model_valid = True
            # When gamma = 1.0 (and even though episodic), straight away starting computation of policies
            # causes numerical issues because of which policy iteration takes
            # extremely long to converge to optimal policy under ill-formed R_hat, T_hat
            # Any value of gamma < 1, like 0.999999 does not cause this problem
            elif np.all(np.delete(self.total_visits, self.exclude, axis=0) > 10):
                self.model_valid = True

    # Does not consider terminal and non-terminal states separately
    # Since there is no need for it
    def get_greedy_policy(self, values):
        # Compute the action value function for pi
        # Take the sum over the final state s'
        Q_pi = np.sum(self.T_hat * (self.R_hat + self.gamma * values), axis=2)
        # Return the greedy policy wrt current action value function Q^{pi}
        pi_greedy = np.argmax(Q_pi, axis=1)
        return pi_greedy

    # Evaluate policy on current model
    def evaluate_policy_on_model(self, pi):
        # The value associated with terminal states is constrained to be = 0.0 in case of episodic tasks
        # coefficient matrix based on bellman's policy evaluation equations
        states = self.nstates
        A = np.identity(states) - self.gamma * self.T_hat[self.states, pi, :]
        # Assign right side b vector as sum of T * R terms
        b = np.sum(self.T_hat[self.states, pi, :] * self.R_hat[self.states, pi, :], axis=1)
        values = np.zeros(states, dtype=np.float32)
        # Check if it is an episodic task, in which case we already know
        # value for terminal states are = 0 (enforce it) (only enter if list non empty)
        if self.exclude:
            # New delete rows and columns corresponding to indices in self.terminal
            A = np.delete(np.delete(A, self.exclude, axis=1), self.exclude, axis=0)
            b = np.delete(b, self.exclude, axis=0)
        # Solve and return Ax = b for non_terminal states, fixed to 0.0 for terminal states
        values[self.non_terminal] = np.linalg.solve(A, b)
        return values

    # Perform MDP planning using Howard's policy iteration algorithm
    # Optimal policy for terminal states is arbitrary since we are using H-PI
    # For faster convergence initialise policy with policy optimal wrt previous model
    def plan(self, init_policy=None):
        # Initialise a random prev and current policy vector
        if init_policy is None:
            pi_prev = np.random.randint(0, self.nactions, self.nstates)
        else:
            # Create a copy since numpy arrays are mutable objects
            pi_prev = init_policy.copy()
        # Perform 1 iteration of Policy Improvement (HPI) to enter while loop
        # Get value function for current policy
        values = self.evaluate_policy_on_model(pi_prev)
        # Attempt to improve policy by evaluating action value functions
        pi_cur = self.get_greedy_policy(values)
        # Begin policy iteration/improvement loop
        while not np.array_equal(pi_prev, pi_cur):
            # Update pi_prev to pi_cur
            pi_prev = pi_cur
            # Get value function for current policy
            values = self.evaluate_policy_on_model(pi_prev)
            # Switch all improvable states as per rules of HPI
            pi_cur = self.get_greedy_policy(values)
        return values, pi_cur
