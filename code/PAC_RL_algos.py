import numpy as np
from random import random
from utils import get_born_state
from model_class import Model


# Model based epsilon-greedy exploration (MBEG)
def MBEG(mdp, budget, eps=0.1):
    # Create a model object with T_hat, R_hat etc
    model = Model(mdp)
    # Init agent state
    cur_state = get_born_state(mdp.born, mdp.nstates)
    for i in range(budget):
        # Choose action as per MBEG algorithm
        rand = random()
        if model.model_valid:
            # MDP plan for optimal policy under current model
            _, pi_optimal = model.plan()
            if rand < eps:
                action = np.random.randint(0, mdp.nactions)
                # print("Random action {0} chosen".format(action))
            else:
                action = pi_optimal[cur_state]
                # print("Greedy action {0} chosen".format(action))
        else:
            action = np.random.randint(0, mdp.nactions)
            # print("Random action {0} chosen".format(action))
        # Query the MDP sample model
        rew, next_state, epi_ended = mdp.sample(cur_state, action)
        model.update(cur_state, action, rew, next_state)
        # print("Next state is {0}".format(next_state))
        cur_state = next_state
        # Be reborn if episode ended
        if epi_ended:
            cur_state = get_born_state(mdp.born, mdp.nstates)
    # print(model.T_hat)
    # print(model.R_hat)
    # Return optimal policy as per learned model
    return model.plan()


# Model based action elimination
def MBAE(mdp, budget):
    return model.plan()
