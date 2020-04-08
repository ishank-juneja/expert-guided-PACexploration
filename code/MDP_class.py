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


# Model to store current understanding of an environment
class Model:
    # Init model using underlying MDP sample model
    def __init__(self, mdp):
        # collect the same state and action space as actual MDP
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
        if self.born > -1:
            return self.born
        else:
            # If -1/any impossible passed, upper limit not included
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
        # Model becomes valid once all non terminal state-action pairs have been reached
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
    def get_greedy_policy(self, values):
        # Compute the action value function for pi
        Q_pi = np.sum(self.T_hat * (self.R_hat + self.gamma * values), axis=2)
        # Return the greedy policy wrt current action values
        pi_greedy = np.argmax(Q_pi, axis=1)
        return pi_greedy

    # Evaluate policy on true MDP or current model
    def evaluate_policy_on_model(self, pi):
        # The value associated with terminal states is constrained to be = 0.0 in case of episodic tasks
        # coefficient matrix based on bellman's policy eval equations
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
    # Optimal policy for terminal states is arbitrary in some sense
    # For faster convergence initialise policy with policy optimal wrt previous model
    def plan(self, init_policy=None):
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
            values = self.evaluate_policy_on_model(pi_prev)
            # Switch all improvable states as per rules of HPI
            pi_cur = self.get_greedy_policy(values)
        return values, pi_cur


class EpsGreedyModel(Model):
    # Initialise like parent model class
    def __init__(self, mdp, eps=0.1):
        super().__init__(mdp)
        # Uniform random exploration probability
        self.eps = eps
        # Policy optimal/greedy with respect to current model
        self.pi_opt = np.random.randint(0, self.nactions, self.nstates)

    # Method to select next action and run an iteration of model learning
    def run_iteration(self, mdp, advice):
        rand = np.random.uniform(0.0, 1.0)
        # If model has become valid by visiting every state-action pair
        if self.model_valid:
            # MDP plan for optimal policy under current model
            v_opt, self.pi_opt = self.plan(init_policy=self.pi_opt)
            # Explore uniformly at random
            if rand < self.eps:
                pac_action = np.random.randint(0, self.nactions)
                # print("Random action {0} chosen".format(action))
            else:
                pac_action = self.pi_opt[self.cur_state]
                # print("Greedy action {0} chosen".format(action))
        # If model not uet valid, choose an action uniformly at random
        else:
            pac_action = np.random.randint(0, self.nactions)
            # print("Random action {0} chosen".format(action))
        # Decide between pac_action and action chosen by advice providing object
        action = advice.seek_advice(pac_action)
        # Query the MDP sample model to see what happens next
        # epi_ended indicates whether current episode has ended
        rew, next_state, epi_ended = mdp.sample(self.cur_state, action)
        # Update model based on what just happened
        self.update(action, rew, next_state)
        # print("Next state is {0}".format(next_state))
        # Update current state
        self.cur_state = next_state
        # Be reborn if episode ended
        if epi_ended:
            self.cur_state = self.get_born_state()
        # print(model.T_hat)
        # print(model.R_hat)
        # Return optimal policy as per learned model, if model_valid


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
