from datetime import datetime
import os

import gym
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

    def __init__(self, game, policy="ddqn", max_mem_size=1e6,
                 lookbehind=4, warm_start=None):
        self.game = game
        self.policy = self._get_policy(policy, warm_start)
        self.memory = ReplayMemory(lookbehind=lookbehind, max_size=max_mem_size)
        self.env = gym.make(game)
        self.env.ale.setInt(b'frame_skip', 4)
        self._lookbehind = lookbehind 

    def _get_policy(self, policy_str, warm_start):
        if policy_str == "ddqn":
            policy = DDQNPolicy(warm_start=warm_start)
        elif policy_str == "dqn":
            policy = DQNPolicy(warm_start=warm_start)
        else:
            raise ValueError("policy must be 'dqn' or 'ddqn'")
        return policy
          
    
    def play(self, n_games=1, epsilon=0.0, monitor_dir=None, verbose=True):
        """ Play multiple games, optionally recording video 
        of gameplay and archiving events in memory."""
        if monitor_dir:
            self.env.monitor.start(monitor_dir, force=True)
        events = []
        for _ in range(n_games):
            events.extend(self._single_play(epsilon))
        if monitor_dir:
            self.env.monitor.close()
        if self.memory:
            self.memory.add(events)
        if verbose:
            msg = "Average Reward Per Game: {}"
            ave_reward = np.sum([x["reward"] for x in events]) / n_games
            print(msg.format(ave_reward))
        return events

    def destroy(self):
        print("Commence destruction of all humans!")

    def choose_action(self, s, epsilon=0.0):
        """Decide whether to explore or exploit past information."""
        if np.random.rand(1) >= epsilon:
            a = np.argmax(self.policy.predict([np.swapaxes(s, 0, 2)]))
        else:
            a = self.random_policy()
        return a

    def random_policy(self, s=None):
        return np.random.randint(0, self.env.action_space.n)

    def _single_play(self, epsilon):
        s_prime = self.env.reset()
        s_prime = process_rgb(s_prime)
        s = [s_prime] * self._lookbehind
        events = []
        while True:
            s = s[-(self._lookbehind - 1):] + [s_prime]
            action = self.choose_action(s, epsilon)
            s_prime, reward, terminal, info = self.env.step(action)
            s_prime = process_rgb(s_prime)
            events.append({"s_prime": s_prime if not terminal else None,
                           "reward": reward,
                           "action": action})
            if terminal:
                break
        return events

    def train(self, ckpt_dir, burnin=1000, games_per_round=100,
              updates_per_round=1000, epsilon=0.1, gamma=0.99,
              rounds_per_epoch=100, n_rounds=1000, batchsize=32):
        self._burnin = burnin
        self._games_per_round = games_per_round
        self._epsilon=epsilon
        self._gamma=gamma
        self._updates_per_round = updates_per_round
        self._rounds_per_epoch = rounds_per_epoch
        self._batchsize = batchsize
        self.play(n_games=burnin, epsilon=1.0)
        for round_i in range(n_rounds):
            self.play(n_games=games_per_round, epsilon=epsilon)
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

if __name__ == "__main__":
    robot = AtariBot(game="Breakout-v0",
                     policy="ddqn",
                     warm_start="models/round_0.ckpt")
    robot.train(ckpt_dir="models", burnin=10, n_rounds=1, updates_per_round=10, games_per_round=10)
