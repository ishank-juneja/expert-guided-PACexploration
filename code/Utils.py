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


def get_born_state(born_state, nstates):
    # Use default born state s_o if the task specifies it
    # Else pick a random state to start with
    if born_state > -1:
        return born_state
    else:
        # If -1/any impossible passed, upper limit not included
        return np.random.randint(0, nstates)


def max_norm_diff(x, y):
    return np.max(np.abs(x - y))


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


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
