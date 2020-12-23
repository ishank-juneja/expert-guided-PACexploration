import numpy as np


# Advice incorporation in the form of a fixed probability
# of trusting external policy alpha
class FixedProbability:
    def __init__(self, model, pi_advice, alpha=0.1):
        # The policy being used as advice called pi^{e} in the report
        self.policy = pi_advice
        # The mdp Model being advised, will be algorithm, specific
        self.mdp_model = model
        # Constant probability of trusting advice
        self.alpha = alpha

    # Function that chooses between actions coming from PAC and external policy
    def seek_advice(self, pac_action):
        rand = np.random.uniform(0.0, 1.0)
        if rand < self.alpha:
            action = self.policy[self.mdp_model.cur_state]
        else:
            action = pac_action
        return action
