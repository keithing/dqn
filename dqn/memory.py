import numpy as np

class ReplayMemory():

    def __init__(self, window=4, max_size=1e5):
        self.episodes = []
        self.window = max(1, window)
        self.max_size = max_size

    def add(self, events):
        self.episodes.append(events)
        if len(self.episodes) > self.max_size:
            self.episodes.pop(0)

    def wipe(self):
        self.events = []

    def sample(self, n):
        xs = np.random.randint(0, high=len(self.episodes), size=n)
        return [self._sample_episode(x) for x in xs]

    def _sample_episode(self, i):
        episode = self.episodes[i]
        j = np.random.randint(self.window + 1, len(episode))
        events = episode[j - self.window - 1:j]
        states = [x["s_prime"] for x in events]
        mem = {"s": states[:-1],
               "s_prime": states[1:],
               "action": events[-1]["action"],
               "reward": events[-1]["reward"]}
        return mem
