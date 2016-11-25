import gym

from dqn.bots import AtariBot
from dqn.policy import DDQNPolicy
from dqn.memory import ReplayMemory

GAME = 'Breakout-v0'

#TODO List params to tune here, eventually migrate this to a readme

if __name__ == "__main__":
    policy = DDQNPolicy()
    memory = ReplayMemory()
    game = gym.make(GAME)
    game.ale.setInt(b'frame_skip', 4)
    robot = AtariBot(policy=policy, memory=memory)
    robot.train(game=game, ckpt_dir="models")
