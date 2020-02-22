from random import randint


def get_born_state(born_state, nstates):
    # Implement Default born state s_o if the task specifies it
    # Else pick a random state to start with
    if born_state > -1:
        return born_state
    else:
        # If -1 passed
        return randint(0, nstates - 1)
