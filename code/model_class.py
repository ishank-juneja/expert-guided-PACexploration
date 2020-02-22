import numpy as np


# Model to store current understanding of an environment
class Model:
    def __init__(self, nstates, nactions):
        # Init containers for eventually learned MDP parameters,
        # we learn the parameters to get a "PAC-MDP" model under the PAC-RL framework
        self.T_hat = np.zeros((nstates, nactions, nstates), dtype=np.float32)
        self.R_hat = np.zeros((nstates, nactions, nstates), dtype=np.float32)
        # Whether model is ready for deployment
        self.model_valid = False
        # Total visits to a state-action pair
        self.total_visits = np.zeros((nstates, nactions), dtype=int)
        # Total transitions made of type s-->a-->s'
        self.total_trans = np.zeros((nstates, nactions, nstates), dtype=int)
        # Total Reward obtained in a transition s,a,s'
        self.total_rew = np.zeros((nstates, nactions, nstates), dtype=np.float32)

    def update_model(self, s, a, r, s_prime):
        # Update visits stats
        self.total_trans[s, a, s_prime] += 1
        self.total_rew[s, a, s_prime] += r
        self.total_visits[s, a, s_prime] += 1
        # Update transition probabilities
        self.T_hat[s, a, :] = self.total_trans[s, a, :]/self.total_visits[s, a]
        # Update reward function
        self.R_hat[s, a, s_prime] = self.total_rew[s, a, s_prime]/self.total_trans[s, a, s_prime]
        if not self.model_valid:
            if np.all(self.total_visits > 0):
                self.model_valid = True

