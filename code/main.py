import numpy as np
from MDP_class import MDP
from PAC_RL_algos import MBEG
import random
import argparse


# Keeps track of agent state and invokes sample model as per the decision of
# PAC-RL algorithms
def main(cmd_args):
    # Set random seed for random library
    if args.rs is not None:
        random.seed(args.rs)
        np.random.seed(args.rs)
    # else no seed
    # Initialise MDP instance by parsing text file with true MDP parameters
    # This MDP object is a sample model, where Transition function and
    # Reward function are hidden/private variables
    mdp = MDP(cmd_args.file_name)
    v_hat, pi_hat = MBEG(mdp, 10000)
    print(v_hat, pi_hat)
    print(mdp.plan())
    return


if __name__ == '__main__':
    # Initialise a parser instance
    parser = argparse.ArgumentParser()
    # Add arguments to the parser 1 at a time
    # The --<string name> indicate optional arguments that follow these special symbols
    # MDP instance file
    parser.add_argument("--mdp", action="store", dest="file_name", type=str)
    # algorithm type
    parser.add_argument("--algorithm", action="store", dest="algo", type=str)
    # Random seed
    parser.add_argument("--seed", action="store", dest="rs", type=int)
    # Reads Command line arguments, converts read arguments to the appropriate data type
    args = parser.parse_args()
    # Call main function
    main(args)
