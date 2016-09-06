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


class ReplayMemory():

    def __init__(self, lookbehind=4, max_size=1e6):
        self.events = []
        self.lookbehind = max(1, lookbehind)
        self.max_size = max_size

    def add(self, s_prime, reward, action, terminal):
        s_prime = None if terminal else process_rgb(s_prime)
        event = {"s_prime": s_prime, "reward": reward, "action": action}
        self.events.append(event)
        if len(self.events) > self.max_size:
            self.events.pop(0)

    def get(self, i):
        event = {}
        events = self.events[i - self.lookbehind - 1:i]
        states = []
        for e in events:
            if e["s_prime"] is None:
                break
            states.append(e["s_prime"])
        if len(states) == (self.lookbehind + 1):
            s = states[:-1]
            s_prime = states[1:]
            event["action"] = events[-1]["action"]
            event["reward"] = events[-1]["reward"]
            event["s"] = s
            event["s_prime"] = s_prime
        return event

    def sample(self, n):
        samples = []
        while len(samples) < n:
            i = np.random.randint(0, len(self.events))
            event = self.get(i)
            if event:
                samples.append(event)
        return samples

    def current_state(self):
        return [x["s_prime"] for x in self.events[-self.lookbehind:]]

    def burnin(self, screen):
        for _ in range(self.lookbehind):
            self.add(s_prime=screen, reward=None, action=None, terminal=False)


