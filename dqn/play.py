import argparse
import logging

import gym
from keras.models import model_from_json
import numpy as np

from dqn import ReplayMemory



def load_model():
    model = model_from_json(open("models/cnn.json").read())
    model.load_weights("models/cnn.h5")
    return(model)


def play(n_times, record=False, epsilon=.9):
    env = gym.make('Breakout-v0')
    if record:
        env.monitor.start('/home/kingersoll/share/breakout', force=True)
    for i_episode in range(n_times):
        single_play(env, epsilon)
    if record:
        env.monitor.close()


def choose_action(env, history, model, epsilon=.9):
    """
    Decide whether to explore or exploit past information.
    """
    explore = lambda: np.random.rand(env.action_space.n)
    exploit = lambda: model.predict(np.array([history]))[0]
    rewards = explore() if np.random.rand(1) < epsilon else exploit()
    return np.argmax(rewards)


def single_play(env, epsilon=.9, model=None, memory=None):
    env.ale.setInt(b'frame_skip', 4)
    mem = memory or ReplayMemory()
    model = model or load_model()
    initial_screen = env.reset()
    mem.burnin(initial_screen)
    rewards = []
    train_labels = []
    while True:

        # Take Action
        action = choose_action(env, mem.current_state(), model, epsilon)
        s_prime, reward, terminal, info = env.step(action)
        mem.add(s_prime, reward, action, terminal)
        rewards.append(reward)

        # Check for breaking condition
        if info:
            print(info)
        if terminal:
            msg = ("Sum Rewards: " + str(np.sum(rewards)))
            print(msg)
            break
    return mem

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_times", default=1, type=int)
    parser.add_argument("-r", "--record", action="store_true")
    parser.add_argument("-e", "--epsilon", default=1.0, type=float)
    return vars(parser.parse_args())


if __name__ == "__main__":
    logging.getLogger("gym").setLevel(logging.WARNING)
    args = parse_args()
    play(**args)
