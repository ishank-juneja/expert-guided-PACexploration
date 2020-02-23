import numpy as np
import sys


# Model to store current understanding of an environment
class Model:
    # Init model
    def __init__(self, mdp):
        self.nstates = mdp.nstates
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
        self.exclude = mdp.terminal
        # Assign 1 probability of coming back to terminal state
        # under any action taken while at terminal state
        for state in self.exclude:
            # Other probabilities = 0 by default
            self.T_hat[state, :, state] = 1.0

    # Update model estimate
    def update(self, s, a, r, s_prime):
        # Update visits stats
        self.total_trans[s, a, s_prime] += 1
        self.total_rew[s, a, s_prime] += r
        self.total_visits[s, a] += 1
        # Update transition probabilities
        self.T_hat[s, a, :] = self.total_trans[s, a, :]/self.total_visits[s, a]
        # Update reward function
        self.R_hat[s, a, s_prime] = self.total_rew[s, a, s_prime]/self.total_trans[s, a, s_prime]
        if not self.model_valid:
            # Can't take any actions from terminal state
            if np.all(np.delete(self.total_visits, self.exclude, axis=0) > 0) and self.gamma < 1.0:
                print("Model became valid")
                self.model_valid = True
            # When gamma = 1, straight away starting computation of policies
            # causes numerical issues because of which policy iteration takes
            # extremely long to converge to optimal policy under ill-formed R_hat, T_hat
            # Any value of gamma < 1, like 0.999999 does not cause this problem
            elif np.all(np.delete(self.total_visits, self.exclude, axis=0) > 10):
                self.model_valid = True

    def get_max_action_value(self, values):
        Q_pi = np.zeros((self.nstates, self.nactions))
        for s in range(self.nstates):
            for a in range(self.nactions):
                Q_pi[s, a] = np.sum(self.T_hat[s, a, :] * (self.R_hat[s, a, :] +
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
            A[s, :] = A[s, :] - self.gamma * self.T_hat[s, pi[s], :states]
        # Assign right side b vector as sum of T * R terms
        b = np.zeros(states)
        for s in range(states):
            b[s] = np.sum(self.T_hat[s, pi[s], :states] * self.R_hat[s, pi[s], :states])
        # Check if it is an episodic task, in which case we already know
        # value for terminal state = 0 (enforce it)
        if self.type == 'episodic':
            A = A[:-1, :-1]
            b = b[:-1]
        # print(A, b)
        # Solve and return Ax = b
        values = np.linalg.solve(A, b)
        # handle it
        if self.type == 'episodic':
            # For last state s = |S| - 1
            values = np.append(values, 0)
        return values

    def evaluate_MBAE_policy(self, pi, explore_terms):
        # Assuming a single terminal state S-1, remove it from
        # policy evaluation to get full rank A matrix
        # Init coeffcient matrix based on diagonal elements having + 1 term
        states = self.nstates
        A = np.identity(states)
        # Assign as per bellman's policy eval equations
        for s in range(states):
            A[s, :] = A[s, :] - self.gamma * self.T_hat[s, pi[s], :states]
        # Assign right side b vector as sum of T * R terms
        b = np.zeros(states)
        for s in range(states):
            # Additional explore term corresposnding to s,pi(s) is added to RHS constant term
            # In solving the linear Bell-man equations --> augmented bellman's equations
            b[s] = np.sum(self.T_hat[s, pi[s], :states] * self.R_hat[s, pi[s], :states]) + explore_terms[s, pi[s]]
        # Check if it is an episodic task, in which case we already know
        # value for terminal state = 0 (enforce it)
        if self.type == 'episodic':
            A = A[:-1, :-1]
            b = b[:-1]
        # print(A, b)
        # Solve and return Ax = b
        values = np.linalg.solve(A, b)
        # handle it
        if self.type == 'episodic':
            # For last state s = |S| - 1
            # Only non zero term in value function will be MBAE explore term
            values = np.append(values, explore_terms[:-1, pi[:-1]])
        return values

    # Perform MDP planning using Howard's policy iteration algo
    # Have assumed that last state is unique terminal state
    def plan(self, init_policy=None, pac_type=None, explore_terms=0):
        # Initialise a random prev and current policy vector
        if init_policy is None:
            pi_prev = np.random.randint(0, self.nactions, self.nstates)
        else:
            pi_prev = init_policy
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
            if pac_type is None:
                values = self.evaluate_policy(pi_prev)
            elif pac_type is 'MBAE':
                values = self.evaluate_MBAE_policy(pi_prev, explore_terms)
            # Attempt to improve policy by evaluating action value functions
            pi_cur = self.get_max_action_value(values)
        return values, pi_cur
