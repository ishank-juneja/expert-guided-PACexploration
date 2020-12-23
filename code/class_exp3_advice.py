# Class for Exp3 advice incorporation using policy roll-outs as proxies for rewards
# The policy roll outs can potentially be quite long but the assumption is that the tasks are episodic
class Exp3:
    def __init__(self, model, pi_advice, mdp, eta=0.1):
        super().__init__(model, pi_advice)
        # MDP sample model to be used to perform rolls outs
        self.sample_model_mdp = mdp
        # Learning rate eta for exp3
        self.eta = eta

    def seek_advice(self, pac_action):
        # dummy method for now, no advice incorporated
        return pac_action

    # Roll out policy pi on episodic mdp sample model mdp
    # Estimate return to associate return with policy
    def roll_out(self, pi, mdp):
        if mdp.type != 'episodic':
            print("Error roll-out attempted on episodic policy")
            exit(-1)
        return None
