from datetime import datetime

import gym
import numpy as np

from dqn import DQN, ReplayMemory
from play import single_play

if __name__ == "__main__":
    dqn = DQN(batchsize=32)
    mem = ReplayMemory(lookbehind=4, max_size=1e6)
    env = gym.make('Breakout-v0')
    cnt = 0
    while True:
        epsilon = max(1 - (cnt / 10000), .1)
        while len(mem.events) < 1000:
            # Create a history to train on
            mem = single_play(env, epsilon, dqn.model, mem)
            print("History size: {}".format(len(mem.events)), flush=True)
        try:
            mem = single_play(env, epsilon, dqn.model, mem)
            dqn.fit(mem, update_target_model = (cnt % 50 == 0))
        except Exception as e:
            print(e)
        if cnt % 50 == 0:
            print("\n\n*****************")
            print("Round: ", str(cnt))
            print("Epsilon: ", epsilon)
            print(datetime.strftime(datetime.now(), "%m/%d %H:%M:%S"))
            print("*****************\n\n")
            dqn.save()
        cnt += 1

