import argparse
import logging

import gym
from keras.models import model_from_json
import numpy as np
from scipy.misc import imresize



def grayscale(rgb):
    """
    http://stackoverflow.com/questions/12201577/
    how-can-i-convert-an-rgb-image-into-grayscale-in-python
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def process_rgb(rbg):
    return imresize(grayscale(rbg), .5)


def load_model():
    model = model_from_json(open("models/cnn.json").read())
    model.load_weights("models/cnn.h5")
    return(model)


def play(n_times, record=False, epsilon=.9):
    env = gym.make('Breakout-v0')
    if record:
        env.monitor.start('monitor/breakout', force=True)
    for i_episode in range(n_times):
        single_play(env, epsilon)
    if record:
        env.monitor.close()


def choose_action(env, history, model, epsilon=.9):
    """
    Decide whether to explore or exploit past information.
    """
    explore = np.random.rand(1) < epsilon
    if explore:
        action = env.action_space.sample()
    else:
        # might get stuck of action 1 never called to send new ball
        X = np.array([history])
        expected_reward = model.predict(X)[0]
        p = np.divide(expected_reward, np.sum(expected_reward))
        action = np.random.choice(len(p), p=p)
    return action


def cache_data(observation, history, reward, action):
    """
    Insert random fraction of data into D for training.
    """
    data = None
    s_prime = history[1:] + [observation]
    data = {"s": history,
            "reward": reward,
            "action": action,
            "s_prime": s_prime}
    return data


def single_play(env, epsilon=.9, model=None):
    model = model or load_model()
    observation = env.reset()
    history = [process_rgb(observation) for _ in range(3)]
    rewards = []
    train_labels = []
    D = []
    while True:
        env.render()

        # Take Action!
        action = choose_action(env, history, model, epsilon)
        observation, reward, done, info = env.step(action)
        observation = process_rgb(observation)
        rewards.append(reward)

        # Check for breaking condition
        if info:
            print(info)
        if done:
            msg = ("Sum Rewards: " + str(np.sum(rewards)))
            print(msg)
            break

        # Update Replay Data
        D.append(cache_data(observation, history, reward, action))

        # Update history
        history.append(observation)
        history.pop(0)
    return D


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
