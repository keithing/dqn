from datetime import datetime

import gym
import numpy as np

from dqn import DQN, ReplayMemory
from play import single_play

import tensorflow as tf

if __name__ == "__main__":
    dqn = DQN(batchsize=32, ckpt_dir="ckpts")
    mem = ReplayMemory(lookbehind=4, max_size=1e6)
    env = gym.make('Breakout-v0')
    cnt = 0
    while True:
        epsilon = max(1 - (cnt / 30000), .1)
        start_new_epoch = cnt % 3000 == 0
        while len(mem.events) < 30000:
            # Create a history to train on
            mem = single_play(env, epsilon, dqn, mem)
            print("History size: {}".format(len(mem.events)), flush=True)
        for _ in range(10):
            mem = single_play(env, epsilon, dqn, mem)
        try:
            dqn.fit(mem, n_updates=500)
        except Exception as e:
            print(e)
        if start_new_epoch:
            print("\n\n*****************")
            print("Round: ", str(cnt))
            print("Epsilon: ", epsilon)
            print(datetime.strftime(datetime.now(), "%m/%d %H:%M:%S"))
            print("*****************\n\n")
            dqn.refresh_target_network()
        cnt += 1

