import numpy as np
from model_eps_greedy import EpsGreedyModel
from model_action_elimination import ActionEliminationModel
from MDP_class import MDP
from class_constant_random_advice import FixedProbability
from Utils import *
import argparse
import pathlib
import sys


# Main thread for experimentation with external knowledge injection
# into MDP exploration problem
# Model based PAC-RL algorithms, to be put to test with and
# without external expert knowledge/some exploration enhancement method
def main(cmd_args, fout):
    # Get list of algorithms to be implemented
    if cmd_args.al is not None:
        # Only run specified algorithms
        algos = args.al
    else:
        # run all available algorithms
        algos = ['eps-greedy', 'action-elimination', 'interval-estimation']
    # Initialize MDP sample model instance by parsing text file with true MDP parameters
    # Transition function and Reward function are hidden/private variables
    mdp_arguments = parse_mdp_file(cmd_args.file_name)
    # use the returned kwargs to create an MDP
    mdp = MDP(**mdp_arguments)
    # Get optimal policy to use as starting point for pi^e
    v_opt, pi_advice = mdp.plan()
    # Take the negative of the reward function to get the negative MDP
    # neg_mdp_arguments = mdp_arguments.copy()
    # neg_mdp_arguments['rewards'] = -1*neg_mdp_arguments['rewards']
    # negative_mdp = MDP(**neg_mdp_arguments)
    # # Use negative mdp to compute good/near optimal policy
    # pi_external = negative_mdp.simpleRPI(pi_optimal, 1)
    horizon = 10000
    # horizons = [100, 200, 500, 1000, 5000, 10000, 20000, 50000]
    # pi_external = np.array([0, 0, 0, 1, 4, 3, 1, 0, 0, 0])
    # alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    for al in algos:
        for rs in range(cmd_args.nseeds):
            # Seed all subsequent random events using this random seed
            np.random.seed(rs)
            print("Currently simulating {0} on {1} with random seed = {2}".format(al, in_name, rs))
            if al == 'eps-greedy':
                model = EpsGreedyModel(mdp, eps=0.1)
                advice = FixedProbability(model, pi_advice, alpha=0.1)
                for t in range(1, horizon + 1):
                    print("Running iteration {0}".format(t))
                    model.run_iteration(mdp, advice)
            elif al == 'action-elimination':
                model = ActionEliminationModel(mdp, eps=0.1, delta=0.1)
                advice = FixedProbability(model, pi_advice)
                for t in range(1, horizon + 1):
                    model.run_iteration(mdp, advice)
            else:
                print("Invalid algorithm {0} encountered, skipped".format(al))
    return


if __name__ == '__main__':
    # Initialise a parser instance
    parser = argparse.ArgumentParser()
    # Add arguments to the parser 1 at a time
    # The --<string name> indicate optional (or required) arguments that follow these special symbols
    # MDP instance file
    parser.add_argument("--mdp", action="store", dest="file_name", type=str, required=True)
    # PAC-RL (or rather PAC-MDP) algorithms stored as a list named algos
    parser.add_argument("--algorithms", action="store", dest="al", type=str, nargs='+', required=False)
    # Random seed
    parser.add_argument("--nseeds", action="store", dest="nseeds", type=int, required=False, default=1)
    # Whether to generate summary file, if passed, True stored in writeFile
    parser.add_argument("--writeFile", action="store_true", dest='writeFile')
    # Reads Command line arguments, converts read arguments to the appropriate data type
    args = parser.parse_args()
    # Get MDP name from file path
    in_name = args.file_name.split('/')[-1].split('.')[0]
    # Check whether output file has to be written
    if args.writeFile:
        # Generate relative paths for output files
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
