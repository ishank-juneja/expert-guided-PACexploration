import numpy as np


# Wrapper class for keeping track of line numbers
# while reading a text file, from stack overflow
class FileLineWrapper(object):
    def __init__(self, f):
        self.f = f
        self.line = 0

    def close(self):
        return self.f.close()

    def readline(self):
        self.line += 1
        return self.f.readline()
    # to allow using in 'with' statements

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def parse_mdp_file(file_name):
    # Open MDP instance file
    fin = FileLineWrapper(open(file_name, 'r'))
    # Read in number of states from line 1 as an int
    nstates = int(fin.readline())
    # Read in number of action types from line 2
    nactions = int(fin.readline())
    # Read in default born state
    born = int(fin.readline())
    # Check if born state lies in range or is -1
    if born < -1 or born > nstates - 1:
        # Throw error
        print("Error in born state on line {0}".format(fin.line))
        exit(-1)
    # Read in reward function into a matrix
    # Init reward matrix, private data member, algos can't access
    rewards = np.zeros((nstates, nactions, nstates))
    # Read nstates x nactions number of lines
    for i in range(nstates * nactions):
        s = i // nactions
        a = i % nactions
        line_data = fin.readline()
        if check_reward(line_data, fin):
            rewards[s][a] = np.fromstring(line_data, dtype=float, sep='\t')
        # Error encountered in parsing line, message printed
        else:
            exit(-1)
    # Read in Transition function into a matrix, private data member, algos can't access
    transitions = np.zeros_like(rewards)
    for i in range(nstates * nactions):
        s = i // nactions
        a = i % nactions
        line_data = fin.readline()
        if check_transition(line_data, fin):
            transitions[s][a] = np.fromstring(line_data, dtype=float, sep='\t')
        # Error in line, message printed
        else:
            exit(-1)
    # Read discount factor
    gamma = float(fin.readline())
    # Read in Problem type as string --> continuing or episodic
    problem_type = fin.readline()[:-1]  # exclude terminal '\n'
    # Close file
    fin.f.close()
    # Check validity
    if problem_type != 'episodic' and problem_type != 'continuing':
        print("Error on line {0}, provided type {1} must be in [episodic, continuing]".format(fin.line, problem_type))
        exit(-1)
    elif gamma > 1.0 or gamma < 0.0:
        print("Discount factor gamma out of range on line {0}".format(fin.line - 1))
        exit(-1)
    elif gamma == 1.0 and problem_type == 'continuing':
        print("Discount factor cannot be 1.0 for continuing MDP: line {0}".format(fin.line - 1))
        exit(-1)
    kwargs = {'nstates': nstates, 'nactions': nactions, 'born': born, 'rewards': rewards,
              'transitions': transitions, 'gamma': gamma, 'problem_type': problem_type}
    return kwargs


def max_norm_diff(x, y):
    return np.max(np.abs(x - y))


# def RepresentsInt(s):
#     try:
#         int(s)
#         return True
#     except ValueError:
#         return False


def print_error(file_line_wrapper):
    print("Error encountered on line {0} while parsing file {1}".format(file_line_wrapper.line,
                                                                        file_line_wrapper.f.name))
    # Exit program, mdp file incorrect
    exit(-1)


# Check if a line is a valid string for reward
# R[s, a] over the next state s'
def check_reward(line_data, file):
    try:
        np.fromstring(line_data, dtype=float, sep='\t')
        return True
    except ValueError:
        print("Invalid characters on line {0} while parsing file {1}".format(file.line, file.f.name))
        return False


# Check if a line is a valid string for the distribution
# T[s, a] over the next state s'
def check_transition(line_data, file):
    # Try the below block of creating a custom
    # discrete distribution random variable with the known distribution
    try:
        # If this fails a ValueError is raised
        dist = np.fromstring(line_data, dtype=float, sep='\t')
        # Even next failing leads to a ValueError
        np.random.choice(len(dist), p=dist)
        return True
    except ValueError:
        print("Problem with transition on line {0} while parsing file {1}".format(file.line, file.f.name))
        return False
