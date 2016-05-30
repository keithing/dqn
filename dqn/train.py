from datetime import datetime

import gym
import numpy as np

from dqn import DQN
from play import single_play


if __name__ == "__main__":
    dqn = DQN(batchsize=50)
    env = gym.make('Breakout-v0')
    cnt = 1
    max_D_size = 300000
    D = []
    while True:
        epsilon = max(1 - (cnt / 10000), .1)
        while len(D) < 1000:
            # Create a history to train on
            D.extend(single_play(env, epsilon, dqn.model))
            print("History size: {}".format(len(D)), flush=True)
        try:
            D.extend(single_play(env, epsilon, dqn.model))
            dqn.fit(D, update_target_model = (cnt % 10 == 0))
            if len(D) >= max_D_size:
                D.pop(0)
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

