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
        if not terminal:
            self.events.append(event)
        if len(self.events) > self.max_size:
            self.events.pop(0)

    def get(self, i):
        """ A missing 's_prime' in events indicates
        a terminal event.  Missing 'action' indicates
        an initial event before an action was taken.
        This returns an event only if the event has
        an associated action, has at least as many
        pre states as lookbehind and doesn't cross
        a terminal event."""
        event = {}
        events = self.events[i - self.lookbehind - 1:i]
        s = [x["s_prime"] for x in events[:-1]]
        s_prime = [x["s_prime"] for x in events[1:]]
        if events and events[-1]["action"]:
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


