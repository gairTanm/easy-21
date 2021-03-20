import numpy as np
from environment import Player
from environment import Action
from environment import State


class Agent(Player):
    def __init__(self, environment, No=100, discount_factor=1):
        Player.__init__(self)
        self.env = environment
        self.No = No
        self.disc_factor = discount_factor
        self.V = np.zeros([self.env.dealer_max_value + 1, self.env.agent_max_value + 1])
        self.wins = 0.0
        self.iterations = 0.0

    def get_clear_tensor(self):
        return np.zeros((self.env.dealer_max_value + 1, self.env.agent_max_value + 1, self.env.actions_count))

    def choose_random_action(self):
        return Action.HIT if np.random.rand() <= 0.5 else Action.STICK

    def choose_best_action(self, s):
        raise NotImplementedError()

    def get_max_action(self, s):
        return 0.0

    def get_value_function(self):
        for i in range(1, self.env.dealer_max_value + 1):
            for j in range(1, self.env.agent_max_value + 1 ):
                s = State(j, i)
                print(s.dealer_sum, s.agent_sum)
                self.V[i][j] = self.get_max_action(s)
        return self.V

    def train(self, steps):
        for e in range(steps):
            pass
        return self.get_value_function()
