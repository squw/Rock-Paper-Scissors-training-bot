import numpy as np
from numpy.random import choice
from numba import jit


class trainer():

    def __init__(self):
        self.TOTAL_CHOICES = 3
        self.REWARD_BANK = np.array([
            [0, -1, 1],
            [1, 0, -1],
            [-1, 1, 0]
        ])
        self.possible_moves = np.arange(self.TOTAL_CHOICES)

        self.player_1_regret_total = np.zeros(self.TOTAL_CHOICES)
        self.player_2_regret_total = np.zeros(self.TOTAL_CHOICES)

        self.player_1_prob_total = np.zeros(self.TOTAL_CHOICES)
        self.player_2_prob_total = np.zeros(self.TOTAL_CHOICES)

    # @jit(nopython=True)
    def get_probabilities(self, regret_total):
        new_regret_total = np.clip(regret_total, a_min=0, a_max=None)
        sum = np.sum(new_regret_total)
        if sum > 0:
            return (new_regret_total / sum)
        else:
            return np.repeat(1/self.TOTAL_CHOICES, self.TOTAL_CHOICES)

    # @jit(nopython=True)
    def get_rewards(self, p1_action, p2_action):
        return self.REWARD_BANK[p1_action, p2_action]

    # @jit(nopython=True)
    def get_action(self, probabilities):
        return choice(self.possible_moves, p=probabilities)

    # @jit(nopython=True)
    def train(self, iteration):
        for i in range(iteration):
            p1_probabilities = self.get_probabilities(
                self.player_1_regret_total)
            p2_probabilities = self.get_probabilities(
                self.player_2_regret_total)
            self.player_1_prob_total += p1_probabilities
            self.player_2_prob_total += p2_probabilities
            p1_action = self.get_action(p1_probabilities)
            p2_action = self.get_action(p2_probabilities)
            p1_reward = self.get_rewards(p1_action, p2_action)
            p2_reward = self.get_rewards(p2_action, p1_action)

            for j in range(self.TOTAL_CHOICES):
                p1_counterfactual_reward = self.get_rewards(j, p2_action)
                p2_counterfactual_reward = self.get_rewards(j, p1_action)
                p1_regret = p1_counterfactual_reward - p1_reward
                p2_regret = p2_counterfactual_reward - p2_reward
                self.player_1_regret_total[j] += p1_regret
                self.player_2_regret_total[j] += p2_regret


def main():
    trainings = trainer()
    trainings.train(1000000)
    p1_prob_avg = trainings.player_1_prob_total / \
        np.sum(trainings.player_1_prob_total)
    p2_prob_avg = trainings.player_2_prob_total / \
        np.sum(trainings.player_2_prob_total)
    print("Player 1's Probabilities are %s" % p1_prob_avg)
    print("Player 2's Probabilities are %s" % p2_prob_avg)


main()
