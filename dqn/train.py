from datetime import datetime

import gym

from dqn import DQN, ReplayMemory
from play import single_play


if __name__ == "__main__":
    dqn = DQN(batchsize=32)
    mem = ReplayMemory(lookbehind=4, max_size=1e6)
    env = gym.make('Breakout-v0')
    i = 0
    while True:
        epsilon = max(1 - (i / 30000), .1)
        start_new_epoch = i % 3000 == 3
        while len(mem.events) < 1000:
            # Create a history to train on
            mem = single_play(env, epsilon, dqn, mem)
            print("History size: {}".format(len(mem.events)), flush=True)
        for _ in range(10):
            mem = single_play(env, epsilon, dqn, mem)
        try:
            dqn.fit(mem, n_updates=50)
        except Exception as e:
            print(e)
        if start_new_epoch:
            print("\n\n*****************")
            print("Round: ", str(i))
            print("Epsilon: ", epsilon)
            print(datetime.strftime(datetime.now(), "%m/%d %H:%M:%S"))
            print("*****************\n\n")
            dqn.checkpoint("ckpts/iter_{}.cpkt".format(i))
            dqn.update_target_network()
        i += 1
