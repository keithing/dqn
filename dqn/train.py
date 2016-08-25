from datetime import datetime

import gym
import numpy as np

from dqn import DQN, ReplayMemory
from play import single_play

import tensorflow as tf

if __name__ == "__main__":
    dqn = DQN(batchsize=32)
    mem = ReplayMemory(lookbehind=4, max_size=1e6)
    env = gym.make('Breakout-v0')
    cnt = 0
    tf_session = tf.Session(graph = dqn.model)
    tf_session.run("init/init")
    while True:
        epsilon = max(1 - (cnt / 30000), .1)
        start_new_epoch = cnt % 200 == 0
        while len(mem.events) < 100:
            # Create a history to train on
            mem = single_play(env, epsilon, dqn.model, mem)
            print("History size: {}".format(len(mem.events)), flush=True)
        for _ in range(2):
            mem = single_play(env, epsilon, dqn.model, mem)
        dqn.fit(mem, tf_session=tf_session,
                update_target_model = start_new_epoch,
                n_updates=500)
        if start_new_epoch:
            print("\n\n*****************")
            print("Round: ", str(cnt))
            print("Epsilon: ", epsilon)
            print(datetime.strftime(datetime.now(), "%m/%d %H:%M:%S"))
            print("*****************\n\n")
            dqn.save()
        cnt += 1

