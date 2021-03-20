import copy
from enum import Enum

import numpy as np


class Color(Enum):
    RED = 0
    BLACK = 0


class Card(object):
    def __init__(self, color=None):
        self.value = self._get_random_value()
        if color == Color.BLACK or color == Color.RED:
            self.color = color
        else:
            self.color = self._get_random_color()

    def _get_random_value(self):
        return np.random.randint(1, 10)

    def _get_random_color(self):
        """
        Color.RED with 1/3 and Color.BLACK with 2/3 probability.
        """
        prob = np.random.rand()
        if prob > 1 / 3.0:
            return Color.BLACK
        else:
            return Color.RED


class Deck(object):
    def take_card(self, color=None):
        return Card(color)


class State(object):
    def __init__(self, agent_sum=0, dealer_sum=0, is_terminal=False):
        self.agent_sum = agent_sum
        self.dealer_sum = dealer_sum
        self.is_terminal = is_terminal


class Action(Enum):
    STICK = 0
    HIT = 1


class Player(object):
    def policy(self):
        raise NotImplementedError


class Dealer(object):
    def policy(self, s):
        if s.dealer_sum >= 17:
            return Action.STICK
        else:
            return Action.HIT


class Environment(object):
    def __init__(self):
        self.dealer = Dealer()
        self.deck = Deck()

        self.agent_max_value = 21
        self.dealer_max_value = 10
        self.actions_count = 2

    def check_bust(self, player_sum):
        return player_sum > self.agent_max_value or player_sum < 1

    def generate_reward_bust(self, s):
        if s.agent_sum > s.dealer_sum:
            return 1
        elif s.agent_sum == s.dealer_sum:
            return 0
        else:
            return -1

    def take_card(self, card_color=None):
        Card = self.deck.take_card(card_color)
        return Card.value if Card.color == Color.BLACK else Card.value * -1

    def dealer_turn(self, s):
        action = None
        while not s.is_terminal:
            action = self.dealer.policy(s)
            if action == Action.HIT:
                s.dealer_sum += self.take_card()
            else:
                break
            s.is_terminal = self.check_bust(s.dealer_sum)
        return s

    def initial_state(self):
        return State(self.take_card(Color.BLACK), self.take_card(Color.BLACK))

    def step(self, s, a):
        r = 0
        next_s = copy.copy(s)

        if a == Action.STICK:
            next_s = self.dealer_turn(s)
            if next_s.is_terminal:
                r = 1
            else:
                next_s.is_terminal = True
                r = self.generate_reward_bust(next_s)
        else:
            next_s.agent_sum += self.take_card(self.deck)
            next_s.is_terminal = self.check_bust(next_s.agent_sum)

            if next_s.is_terminal:
                r = -1
        return next_s, r


environment = Environment()