import numpy as np

class ReplayMemory():

    def __init__(self, lookbehind=4, max_size=1e6):
        self.events = []
        self.lookbehind = max(1, lookbehind)
        self.max_size = max_size

    def add(self, events):
        self.events.extend(events)
        if len(self.events) > self.max_size:
            self.events.pop(0)

    def get(self, i):
        mem = {}
        events = self.events[i - self.lookbehind - 1:i]
        states = [x["s_prime"] for x in events]
        missing = any([x is None for x in states])
        mem["s"] = states[:-1]
        mem["s_prime"] = states[1:]
        mem["action"] = events[-1]["action"]
        mem["reward"] = events[-1]["reward"]
        return mem, missing

    def sample(self, n):
        samples = []
        while len(samples) < n:
            i = np.random.randint(self.lookbehind + 1, len(self.events))
            event, missing = self.get(i)
            if not missing:
                samples.append(event)
        return samples
