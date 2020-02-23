import numpy as np
from MDP_class import MDP
from PAC_RL_algos import *
import random
import argparse
import matplotlib.pyplot as plt


# Keeps track of agent state and invokes sample model as per the decision of
# PAC-RL algorithms
def main(cmd_args):
    # Set random seed for random library
    # if args.rs is not None:
    #     random.seed(args.rs)
    #     np.random.seed(args.rs)
    # else no seed
    # Initialise MDP instance by parsing text file with true MDP parameters
    # This MDP object is a sample model, where Transition function and
    # Reward function are hidden/private variables
    mdp = MDP(cmd_args.file_name)
    # Obtain true value function for MDP
    values, pi_star = mdp.plan()
    horizons = [100, 200, 500, 1000, 5000, 10000, 20000, 50000]
    pi_external = np.array([0, 0, 0, 1, 4, 3, 1, 0, 0, 0])
    # alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    nseeds = 15
    dev_expert = np.zeros((len(horizons), nseeds))
    dev_vanilla = np.zeros((len(horizons), nseeds))
    for rs in range(nseeds):
        print("Random Seed = {0}".format(rs))
        true_v_lst, (v_hat, pi_hat) = MBEG(mdp, 50000, eps=0.1, alpha=0.1, pi_external=pi_external, horizons=horizons)
        for horizon, value in enumerate(true_v_lst):
            dev_expert[horizon, rs] = np.max(np.abs(value - values))
        true_v_lst, (v_hat, pi_hat) = MBEG(mdp, 50000, eps=0.1, horizons=horizons)
        for horizon, value in enumerate(true_v_lst):
            dev_vanilla[horizon, rs] = np.max(np.abs(value - values))
    eps_array_expert = np.mean(dev_expert, axis=1)
    eps_array_vanilla = np.mean(dev_vanilla, axis=1)
    plt.plot(horizons, eps_array_expert, color='b')
    plt.plot(horizons, eps_array_vanilla, color='r')
    plt.xlabel("Horizon - T")
    plt.ylabel("Diff. b/w optimal and learned policy")
    plt.title("Effect of external policy on exploration")
    plt.legend(["With Expert", "Without Expert"])
    plt.savefig("../figures/mdp10.png")
    plt.show()
    plt.close()
    # mean_horizon = np.zeros((nseeds, len(alphas)), dtype=np.float32)
    # for rs in range(nseeds):
    #     print("Random Seed = {0}".format(rs))
    #     for it, alpha in enumerate(alphas):
    #         random.seed(rs)
    #         np.random.seed(rs)
    #         horizon, (v_hat1, pi_hat1) = MBEG(mdp, 500000, eps=0.1, alpha=alpha, pi_external=pi_external)
    #         mean_horizon[rs, it] += horizon
    # mean_horizon = np.mean(mean_horizon, axis=0)
    # plt.loglog(alphas, mean_horizon, color='b')
    # plt.xlabel("External Policy trust probability")
    # #plt.xticks(alphas)
    # plt.ylabel("Horizon to learn optimal policy")
    # plt.title("Effect of external policy on exploration")
    # plt.savefig("../figures/riverSwim.png")
    # plt.show()
    # plt.close()
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
