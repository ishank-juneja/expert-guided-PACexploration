from model_class import Model
import numpy as np


# Model based action elimination, an eps-delta PAC-RL algorithm
# Even-Dar et al. 2004, 2006, JMLR
class ActionEliminationModel(Model):
    def __init__(self, mdp, eps=0.1, delta=0.1):
        # Initialise all parameters of
        super().__init__(mdp)
        # Epsilon-delta PAC parameters
        self.eps = eps
        self.delta = delta
        # And some more AE specific parameters
        self.rmax = mdp.rmax
        self.vmax = self.rmax / (1 - self.gamma)
        # List to keep track of eliminated and active states
        # Initially all states are active, AE eliminates them over time
        self.active = [list(range(self.nactions)) for i in range(self.nstates)]
        # Upper and lower confidence policies for current model
        self.pi_upper = np.random.randint(0, self.nactions, self.nstates)
        self.pi_lower = np.random.randint(0, self.nactions, self.nstates)

    # For internal use, gets explore terms based on internal state
    def get_explore_terms(self):
        # Get explore terms as defined in the paper
        explore_terms = self.vmax * np.sqrt(
                    np.log((self.time ** 2) * self.nstates * self.nactions / self.delta) / self.total_visits)
        return explore_terms

    def run_iteration(self, mdp, advice):
        # Exploration bonus terms for every (s, a)
        explore_terms = self.get_explore_terms()
        if self.model_valid:
            # Choose any random non-eliminated action as per AE algorithm
            pac_action = np.random.choice(self.active[self.cur_state])
            # Get V_UCB, upper confidence bound on values
            v_upper, self.pi_upper = self.plan_upper_lower(explore_terms, init_policy=self.pi_upper)
            # Get V_LCB, lower confidence bound on values, same as above except explore terms are -ve
            v_lower, self.pi_lower = self.plan_upper_lower(-1*explore_terms, init_policy=self.pi_lower)
            Q_upper = np.sum(self.T_hat * (self.R_hat + self.gamma * v_upper), axis=2) + explore_terms
            Q_lower = np.sum(self.T_hat * (self.R_hat + self.gamma * v_lower), axis=2) - explore_terms
            # Eliminate sub-optimal actions for all states
            for s in range(self.nstates):
                # For each state iterate over non eliminated actions
                for a in self.active[s]:
                    if Q_upper[s, a] < v_lower[s]:
                        self.active[s].remove(a)
            # Check if eps-delta pac criteria has converged
            converged = True
            for s in range(self.nstates):
                for a in self.active[s]:
                    # print(Q_upper[s, a], Q_lower[s, a])
                    if abs(Q_upper[s, a] - Q_lower[s, a]) > self.eps * (1 - self.gamma) / 2:
                        converged = False
                        break
            if converged:
                print("Converged")
        else:
            # select action uniformly at random
            pac_action = np.random.randint(0, self.nactions)
        # Decide between pac_action and action chosen by advice providing object
        action = advice.seek_advice(pac_action)
        # Query the MDP sample model
        rew, next_state, epi_ended = mdp.sample(self.cur_state, action)
        # Update agent's model of MDP
        self.update(action, rew, next_state)
        # print("Next state is {0}".format(next_state))
        self.cur_state = next_state
        # Be reborn if episode ended
        if epi_ended:
            self.cur_state = self.get_born_state()
        # print(model.T_hat)
        # print(model.R_hat)
        print(self.active)
        # Return optimal policy as per learned model, if model_valid
        if self.model_valid:
            return np.argmax(Q_lower, axis=1)
        else:
            print("Exploration unsuccessful, model not learnt")
            exit(-1)

    def evaluate_upper_lower(self, pi, explore_terms):
        # The value associated with terminal states is constrained to be = 0.0 in case of episodic tasks
        # coefficient matrix based on bellman's policy eval equations
        states = self.nstates
        A = np.identity(states) - self.gamma * self.T_hat[self.states, pi, :]
        # Assign right side b vector as sum of T * R terms
        b = np.sum(self.T_hat[self.states, pi, :] * self.R_hat[self.states, pi, :], axis=1)
        # Additional explore term corresponding to (s, pi(s)) is added to RHS
        # In solving the linear Bell-man equations --> augmented bellman's equations
        b += explore_terms[self.states, pi]
        # Array to hold values
        values = np.zeros(states, dtype=np.float32)
        # Check if it is an episodic task, in which case we already know
        # value for terminal states are = 0 (enforce it) (only enter if list non empty)
        if self.exclude:
            # New delete rows and columns corresponding to indices in self.terminal
            A = np.delete(np.delete(A, self.exclude, axis=1), self.exclude, axis=0)
            b = np.delete(b, self.exclude, axis=0)
        # Solve and return Ax = b for non_terminal states, fixed to 0.0 for terminal states
        values[self.non_terminal] = np.linalg.solve(A, b)
        # Manually add the exploration terms to the value of the terminal states
        # For terminal states Only non zero term in value function will be MBAE explore term
        values[self.exclude] = explore_terms[self.exclude, pi[self.exclude]]
        return values

    # Perform MDP planning using Howard's policy iteration algorithm
    # Optimal policy for terminal states is arbitrary in some sense
    # For faster convergence initialise policy with policy optimal wrt previous model
    def plan_upper_lower(self, explore_terms, init_policy=None):
        # Initialise a random prev and current policy vector
        if init_policy is None:
            pi_prev = np.random.randint(0, self.nactions, self.nstates)
        else:
            pi_prev = init_policy.copy()
        # Perform 1 iteration of HPI to enter while loop
        # Get value function for current policy
        values = self.evaluate_policy_on_model(pi_prev)
        # Attempt to improve policy by evaluating action value functions
        pi_cur = self.get_greedy_policy(values)
        # Begin policy iteration/improvement loop
        while not np.array_equal(pi_prev, pi_cur):
            # Update pi_prev to pi_cur
            pi_prev = pi_cur
            # Get value function for current policy
            values = self.evaluate_upper_lower(pi_prev, explore_terms)
            # Switch all improvable states as per rules of HPI
            pi_cur = self.get_greedy_policy(values)
        return values, pi_cur
