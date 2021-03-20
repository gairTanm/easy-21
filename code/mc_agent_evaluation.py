from agent import *
from environment import *
from plotting import *


class MCAgentEvaluation(Agent):

    def __init__(self, environment, No=100, discount_factor=1):
        Agent.__init__(self, environment, No, discount_factor)

        self.N = self.get_clear_tensor()

        self.G_s = np.zeros([self.env.dealer_max_value + 1, self.env.agent_max_value + 1])

    def get_value_function(self):
        return self.V

    def predict(self, episode):
        j = 0
        for s, a, _ in episode:
            d_sum = s.dealer_sum
            a_sum = s.agent_sum
            Gt = sum([x[2] * (self.disc_factor ** i) for i, x in enumerate(episode[j:])])
            self.G_s[d_sum][a_sum] += Gt
            self.V[d_sum][a_sum] = self.G_s[d_sum][a_sum] / sum(self.N[s.dealer_sum, s.agent_sum, :])
            j += 1

    def policy(self, s):
        if s.agent_sum >= 17:
            action = Action.STICK
        else:
            action = Action.HIT

        self.N[s.dealer_sum][s.agent_sum][action.value] += 1
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

            self.iterations += 1
            if e % 10000 == 0 and e != 0:
                print("Episode: %d" % e)
            self.predict(episode)

        return self.get_value_function()

environment = Environment()
agent_eval = MCAgentEvaluation(environment)
agent_eval.train(100000)

plot_value_function(agent_eval, title='Value function: acting like the dealer (MC policy evaluation)')
agent_eval = MCAgentEvaluation(environment)