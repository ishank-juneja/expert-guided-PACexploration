from random import randint


# Class to implement riverSwim episodic MDP
class WindyGridWorld:
    # Constructor to init grid world parameters
    def __init__(self, shape, start, goal, wind, stochastic):
        # Read in size of grid world as a tuple
        self.shape = shape
        # Read in default born/start state
        self.start = start
        # Read in goal state
        self.goal = goal
        # Read in the wind strenth for each col
        self.wind = wind
        # Reset current state to start
        self.cur_row = self.start[0]
        self.cur_col = self.start[1]
        self.stochastic = stochastic

    def reset(self):
        # Reset current state to start
        self.cur_row = self.start[0]
        self.cur_col = self.start[1]
        return

    # Action number vs outcome mapping
    # 0 = N, 1 = S, 2 = E, 3 = W
    # 4 = NE, 5 = SE, 6 = SW, 7 = NW
    def update_state(self, action):
        # Shift upward as per wind strength for current column
        # Wind can only push up in this problem
        cur_wind = self.wind[self.cur_col]
        if self.stochastic:
            cur_wind += randint(-1, 1)
        # Fixed negative reward for every step
        rew = -1
        if action == 0:
            # Up
            self.cur_row += 1
        elif action == 1:
            # Down
            self.cur_row += -1
        elif action == 2:
            # Right
            self.cur_col += 1
        elif action == 3:
            # Left
            self.cur_col += -1
        elif action == 4:
            self.cur_row += 1
            self.cur_col += 1
        elif action == 5:
            self.cur_row += -1
            self.cur_col += 1
        elif action == 6:
            self.cur_row += -1
            self.cur_col += -1
        elif action == 7:
            self.cur_row += 1
            self.cur_col += -1
        elif action == 8:
            # No update in row/col
            self.cur_row = self.cur_row
        else:
            print("Incorrect action, aborting")
            exit(0)
        # Add effect of wind
        self.cur_row = max(min(self.cur_row + cur_wind, self.shape[0] - 1), 0)
        self.cur_col = max(min(self.cur_col, self.shape[1] - 1), 0)
        # Episode ended
        if self.cur_row == self.goal[0] and self.cur_col == self.goal[1]:
            rew = 0
        # Return reward and new state to main
        return rew, (self.cur_row, self.cur_col)