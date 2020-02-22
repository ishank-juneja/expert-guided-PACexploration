import numpy as np
from random import random
from utils import get_born_state


# Model based epsilon-greedy exploration (MBEG)
def MBEG(mdp, born_state, eps=0.1):

    # Init agent state
    cur_state = get_born_state(born_state, mdp.nstates)
    # Choose action as per MBEG rules
    action = 0
    # Query the MDP sample model
    rew, next_state, epi_ended = mdp.sample(cur_state)
    update_model(cur_state, action, rew, next_state)
    return T_hat, R_hat
