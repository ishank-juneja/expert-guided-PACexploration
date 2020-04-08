import numpy as np


# Base class for advice incorporation in the form of a policy
class Advice:
    def __init__(self, model, pi_advice):
        # The policy being used as advice
        self.policy = pi_advice
        # The mdp Model being advised
        self.mdp_model = model

    # Choose between own action and pac_action using scheme specific weights
    def seek_advice(self, pac_action):
        # dummy method, no advice incorporated
        return pac_action


# Advice incorporation in the form of a fixed probability
# of trusting external policy alpha
class FixedProbability(Advice):
    def __init__(self, model, pi_advice, alpha):
        super().__init__(model, pi_advice)
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


# Class for exp3 advice incorporation using policy roll-outs as rewards
class Exp3(Advice):
    def __init__(self, model, pi_advice, mdp, eta=0.1):
        super().__init__(model, pi_advice)
        # MDP sample model to be used to perform rolls outs
        self.sample_model_mdp = mdp
        # Learning rate eta for exp3
        self.eta = eta

    # Roll out policy pi on episodic mdp sample model mdp
    # Estimate return to associate return with policy
    def roll_out(self, pi, mdp):
        if mdp.type != 'episodic':
            print("Error roll-out attempted on episodic policy")
            exit(-1)
        return None
