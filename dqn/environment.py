import argparse
import logging

import gym
import numpy as np
from scipy.misc import imresize
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


class Atari():
    
    def __init__(self, policy, monitor_dir=None, game="Breakout-v0",
                 lookbehind=4):
        self.policy = policy
        self.monitor_dir = monitor_dir
        self.game = game
        self.lookbehind = lookbehind
        self.env = gym.make(self.game)
        self.env.ale.setInt(b'frame_skip', 4)

    def play(self, n_games=1, epsilon=0.0):
        if self.monitor_dir:
            self.env.monitor.start(monitor_dir, force=True)
        events = []
        for _ in range(n_games):
            events.extend(self._single_play(epsilon))
        if self.monitor_dir:
            self.env.monitor.close()
        return events

    def choose_action(self, s, epsilon):
        """Decide whether to explore or exploit past information."""
        if np.random.rand(1) < epsilon:
            a = np.random.randint(0, self.env.action_space.n)
        else:
            a = np.argmax(self.policy.predict([np.swapaxes(s, 0, 2)]))
        return a


    def _single_play(self, epsilon):
        s_prime = self.env.reset()
        s_prime = process_rgb(s_prime)
        s = [s_prime] * self.lookbehind
        events = []
        while True:
            s = s[-(self.lookbehind - 1):] + [s_prime]
            action = self.choose_action(s, epsilon)
            s_prime, reward, terminal, info = self.env.step(action)
            s_prime = process_rgb(s_prime)
            events.append({"s_prime": s_prime if not terminal else None,
                           "reward": reward,
                           "action": action})
            if terminal:
                break
        return events
