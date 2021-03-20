from environment import *
from plotting import *
from agent import *
import random


class MCAgentControl(Agent):

    def __init__(self, environment, No=100, discount_factor=1):
        Agent.__init__(self, environment, No, discount_factor)
        self.Q = self.get_clear_tensor()
        self.N = self.get_clear_tensor()

    def get_alpha(self, s, a):
        return 1.0 / (self.N[s.dealer_sum][s.agent_sum][a.value])

    def get_e(self, s):
        return self.No / (self.No + sum(self.N[s.dealer_sum, s.agent_sum, :]) * 1.0)

    def get_max_action(self, s):
        return np.max(self.Q[s.dealer_sum][s.agent_sum])

    def choose_best_action(self, s):
        return Action.HIT if np.argmax(self.Q[s.dealer_sum][s.agent_sum]) >= 1 else Action.STICK

    def control(self, episode):
        j = 0
        for s, a, _ in episode:
            d_sum = s.dealer_sum
            a_sum = s.agent_sum

            Gt = sum([x[2] * (self.disc_factor ** i) for i, x in enumerate(episode[j:])])

            self.N[d_sum][a_sum][a.value] += 1

            error = Gt - self.Q[d_sum][a_sum][a.value]
            self.Q[d_sum][a_sum][a.value] += self.get_alpha(s, a) * error

            j += 1

    def policy(self, s):
        r = random.random()
        if r <= self.get_e(s):
            action = self.choose_random_action()
        else:
            action = self.choose_best_action(s)

        return action

    def train(self, steps):
        for e in range(steps):
            episode = []
            s = self.env.initial_state()
            while not s.is_terminal:
                a = self.policy(s)
                next_s, r = self.env.step(copy.copy(s), a)
                episode.append((s, a, r))
                s = next_s

            if e % 10000 == 0 and self.iterations > 0:
                print("Episode: %d, score: %f" % (e, (float(self.wins) / (self.iterations) * 100.0)))
            self.iterations += 1

            if r == 1:
                self.wins += 1

            self.control(episode)

        return self.get_value_function()

environment = Environment()
mc_agent = MCAgentControl(environment)
mc_agent.train(1000000)

plot_value_function(mc_agent, title="MC Control Value function No=100")
