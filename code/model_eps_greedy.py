from model_class import Model
import numpy as np


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
