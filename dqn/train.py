from datetime import datetime

import gym
import numpy as np

from dqn import DQN
from play import single_play


if __name__ == "__main__":
    dqn = DQN(batchsize=50, n_samples=1000)
    env = gym.make('Breakout-v0')
    cnt = 1
    max_D_size = 150000
    D = []
    while True:
        epsilon = max(1 - (cnt / 100000), .1)
        while len(D) < max_D_size:
            # Burn in a history to train on
            D.extend(single_play(env, epsilon, dqn.model))
        try:
            D.extend(single_play(env, epsilon, dqn.model))
            np.random.shuffle(D)
            dqn.fit(D)
            D = D[:max_D_size]
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

