import numpy as np
from MDP_class import MDP
import PACpolicies
from Utils import *
import argparse
import pathlib
import sys


# Main thread for experimentation with external knowledge injection
# into MDP exploration problem
# Model based PAC-Rl algorithms, to be put to test with and
# without external expert knowledge/some exploration enhancement method
def main(cmd_args, fout):
    # Initialise MDP instance by parsing text file with true MDP parameters
    # This MDP object is purely a sample model
    # Transition function and Reward function are hidden/private variables
    mdp = MDP(cmd_args.file_name)
    # Get list of algorithms to be implemented
    if cmd_args.al is not None:
        # Only run specified algorithms
        algos = args.al
    else:
        # run all available algorithms
        algos = ['mb_eps_greedy', 'mb_action_elimination']
    # Obtain true value function for MDP
    values, pi_star = mdp.plan()
    for i in range(len(values)):
        print(values[i])
    print(pi_star)
    # horizons = [100, 200, 500, 1000, 5000, 10000, 20000, 50000]
    # pi_external = np.array([0, 0, 0, 1, 4, 3, 1, 0, 0, 0])
    # alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    # dev_expert = np.zeros((len(horizons), args.nseeds))
    # dev_vanilla = np.zeros((len(horizons), args.nseeds))

    # for al in algos:
    #     # Get the method associated with the al
    #     explore_method = getattr(PACpolicies, al)
    #     for rs in range(args.nseeds):
    #         # Seed all subsequent random events using this random seed
    #         np.random.seed(rs)
    #         print("Currently simulating {0} on MDP {1} Random Seed = {2}".format(al, in_name, rs))
    #         true_v_lst, (v_hat, pi_hat) = explore_method(mdp, 50000, eps=0.1, horizons=horizons)
    #         for horizon, value in enumerate(true_v_lst):
    #             dev_vanilla[horizon, rs] = max_norm_diff(value, values)
    #     # Write algorithm wise data to file
    #     eps_array_expert = np.mean(dev_expert, axis=1)
    #     eps_array_vanilla = np.mean(dev_vanilla, axis=1)
    #         # Write data from current run to file
    #         # Record data at intervals of STEP in file
    #         if t % STEP == 0:
    #             fout.write(
    #                 "{0}, {1}, {2}, {3}, {4:.2f}, {5:.2f}, {6:.2f}, {7:.2f}\n".format(al, rs, eps, t, REG,
    #                                                                                   mu_max * t - REW, AoI_REG,
    #                                                                                   AoI_cum - t / mu_max))
    # plt.plot(horizons, eps_array_expert, color='b')
    # plt.plot(horizons, eps_array_vanilla, color='r')
    # plt.xlabel("Horizon - T")
    # plt.ylabel("Diff. b/w optimal and learned policy")
    # plt.title("Effect of external policy on exploration")
    # plt.legend(["With Expert", "Without Expert"])
    # plt.savefig("../figures/mdp10.png")
    # plt.show()
    # plt.close()
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
    parser.add_argument("--mdp", action="store", dest="file_name", type=str, required=True)
    # PAC-RL (or rather PAC MDP) algorithms stored as a list named algos
    parser.add_argument("--algorithms", action="store", dest="al", type=str, nargs='+', required=False)
    # Random seed
    parser.add_argument("--nseeds", action="store", dest="rs", type=int, required=False, default=1)
    # Whether to generate summary file, if passed, True stored in writeFile
    parser.add_argument("--writeFile", action="store_true", dest='writeFile')
    # Reads Command line arguments, converts read arguments to the appropriate data type
    args = parser.parse_args()
    # Check whether output file has to be written
    if args.writeFile:
        # Generate relative paths for output files
        in_name = args.file_name.split('/')[-1].split('.')[0]
        out_folder = '../results/' + in_name
        # Create folder to save results if it doesn't already exist
        pathlib.Path(out_folder).mkdir(parents=False, exist_ok=True)
        # Output summary txt file path
        out_file = out_folder + in_name + '-out.txt'
        # File output object in overwrite mode
        out_stream = open(out_file, 'w')
    else:
        # Else push results to std output
        out_stream = sys.stdout
    # Call main function
    main(args, out_stream)
