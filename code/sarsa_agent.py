from environment import *
from plotting import *
from agent import *
import random


class SarsaAgent(Agent):

    def __init__(self, environment, No=100, discount_factor=1, _lambda=1):
        Agent.__init__(self, environment, No, discount_factor)
        self._lambda = _lambda
        self.E = self.get_clear_tensor()
        self.Q = self.get_clear_tensor()
        self.N = self.get_clear_tensor()

    def get_q(self, s, a):
        return self.Q[s.dealer_sum][s.agent_sum][a.value]

    def get_alpha(self, s, a):
        return 1.0 / (self.N[s.dealer_sum][s.agent_sum][a.value])

    def get_e(self, s):
        return self.No / (self.No + sum(self.N[s.dealer_sum, s.agent_sum, :]) * 1.0)

    def get_max_action(self, s):
        return np.max(self.Q[s.dealer_sum][s.agent_sum])

    def choose_best_action(self, s):
        return Action.HIT if np.argmax(self.Q[s.dealer_sum][s.agent_sum]) == 1 else Action.STICK

    def policy(self, s):
        r = random.random()
        if r <= self.get_e(s):
            action = self.choose_random_action()
        else:
            action = self.choose_best_action(s)

        self.N[s.dealer_sum][s.agent_sum][action.value] += 1
        return action

    def train(self, steps):
        for e in range(steps):
            self.E = self.get_clear_tensor()
            s = self.env.initial_state()
            a = self.policy(s)
            next_a = a
            while not s.is_terminal:
                next_s, r = self.env.step(copy.copy(s), a)

                q = self.get_q(s, a)

                if not next_s.is_terminal:
                    next_a = self.policy(next_s)
                    q_next = self.get_q(next_s, next_a)
                    delta = r + (q_next - q) * self._lambda
                else:
                    delta = r - q * self._lambda

                self.E[s.dealer_sum][s.agent_sum][a.value] += 1
                alpha = self.get_alpha(s, a)
                update_q = alpha * delta * self.E
                self.Q += update_q
                self.E *= (self.disc_factor * self._lambda)
                s = next_s
                a = next_a

            if e % 10000 == 0 and e != 0:
                print("Episode: %d, score: %f" % (e, (float(self.wins) / self.iterations) * 100))
            self.iterations += 1
            if r == 1:
                self.wins += 1

        return self.get_value_function()


environment = Environment()

sarsa_agent = SarsaAgent(environment)
sarsa_agent.train(2000000)

plot_value_function(sarsa_agent, title="Sarsa No=100 lambda = 1")
