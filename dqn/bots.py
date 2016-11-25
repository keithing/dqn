from collections import deque
from datetime import datetime
import os

import numpy as np
from scipy.misc import imresize
from dqn.memory import ReplayMemory
from dqn.policy import DDQNPolicy, DQNPolicy


def grayscale(rgb):
    """
    http://stackoverflow.com/questions/12201577/
    how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def process_rgb(rbg):
    return imresize(grayscale(rbg), (84, 84))


class AtariBot:

    def __init__(self, policy, memory):
        self.policy = policy
        self.memory = memory

    def destroy(self):
        print("Commencing destruction of all humans!")

    def choose_action(self, s, game, epsilon=0.0):
        """Decide whether to explore or exploit past information."""
        if np.random.rand(1) >= epsilon:
            a = np.argmax(self.policy.predict([np.swapaxes(s, 0, 2)]))
        else:
            a = game.action_space.sample()
        return a

    def play(self, game, n_games=1, epsilon=0.0, monitor_dir=None):
        """ Play multiple games, optionally recording video
        of gameplay and archiving events in memory."""
        if monitor_dir:
            game.monitor.start(monitor_dir, force=True)
        for n in range(n_games):
            episode = self._single_play(game, epsilon)
            self.memory.add(episode)
        if monitor_dir:
            game.monitor.close()

    def _single_play(self, game, epsilon):
        episode = []
        game_over = False
        s_prime = process_rgb(game.reset())
        s = deque(maxlen=self.policy.window)
        for _ in range(self.policy.window):
            s.append(s_prime)
        while not game_over:
            action = self.choose_action(s, game, epsilon)
            raw_s_prime, reward, game_over, _ = game.step(action)
            s_prime = process_rgb(raw_s_prime)
            s.append(s_prime)
            episode.append({"s_prime": s_prime,
                            "reward": reward,
                            "action": action})
        return episode

    def train(self, game, ckpt_dir, burnin=1000, games_per_round=100,
              updates_per_round=1000, epsilon=0.1, gamma=0.99,
              rounds_per_epoch=100, n_rounds=1000, batchsize=32):
        self.play(game=game, n_games=burnin, epsilon=1.0)
        for round_i in range(n_rounds):
            self.play(game=game, n_games=games_per_round, epsilon=epsilon)
            self.policy.fit(self.memory, updates_per_round, batchsize, gamma)
            if round_i % rounds_per_epoch == 0:
                self._new_train_epoch(ckpt_dir, round_i, epsilon)

    def _new_train_epoch(self, ckpt_dir, round_i, epsilon, verbose=True):
        self.policy.checkpoint(
            os.path.join(ckpt_dir, "round_{}.ckpt".format(round_i)))
        self.policy.update_target_network()
        if verbose:
            msg = "\n\nRound: {}\nEpsilon: {}\nTime: {}\n\n"
            now = datetime.strftime(datetime.now(), "%m/%d %H:%M:%S")
            print(msg.format(round_i, epsilon, now))
