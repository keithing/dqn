from datetime import datetime

import gym

from dqn import DQN, ReplayMemory
from play import single_play


if __name__ == "__main__":
    dqn = DQN(batchsize=32)
    mem = ReplayMemory(lookbehind=4, max_size=500000)
    env = gym.make('Breakout-v0')
    i = 0
    epoch = 0
    while True:
        epsilon = max(1 - (i / 50000), .1)
        start_new_epoch = i % 5000 == 0
        while len(mem.events) < 30000:
            # Create a history to train on
            mem = single_play(env, epsilon, dqn, mem)
            print("History size: {}".format(len(mem.events)), flush=True)
        try:
            for _ in range(10):
                mem = single_play(env, epsilon, dqn, mem)
            dqn.fit(mem, n_updates=200)
        except Exception as e:
            print(e)
        if start_new_epoch:
            epoch += 1
            print("\n\n*****************")
            print("Round: ", str(i))
            print("Epsilon: ", epsilon)
            print(datetime.strftime(datetime.now(), "%m/%d %H:%M:%S"))
            print("*****************\n\n")
            dqn.checkpoint("models/epoch_{}.cpkt".format(epoch))
            dqn.update_target_network()
        i += 1
