from datetime import datetime

import gym
import numpy as np

from dqn import Atari, DQN, ReplayMemory

def train_dqn(batchisize=32, lookbehind=4, max_memory_size=1e6,
              burnin=1000, games_per_round=100,
              updates_per_round=1000, eps_dec_per_round=.0005,
              rounds_per_epoch=100):

    dqn = DQN(batchsize=32)
    mem = ReplayMemory(lookbehind=4, max_size=max_memory_size)
    env = Atari(policy=dqn)
    i = 0
    epoch = 0
    mem.add(env.play(n_games=burnin, epsilon=1.0))
    while True:
        epsilon = max(1 - (i * eps_dec_per_round), .05)
        events = env.play(n_games=games_per_round, epsilon=epsilon)
        mem.add(events)
        dqn.fit(mem, n_updates=updates_per_round)
        print(np.sum([x["reward"] for x in events]) / 10)
        if i % rounds_per_epoch == 0:
            epoch += 1
            print("\n\n*****************")
            print("Round: ", str(i))
            print("Epsilon: ", epsilon)
            print(datetime.strftime(datetime.now(), "%m/%d %H:%M:%S"))
            print("*****************\n\n")
            dqn.checkpoint("models/epoch_{}.cpkt".format(epoch))
            dqn.update_target_network()
        i += 1

if __name__ == "__main__":
    train_dqn(burnin=10)
