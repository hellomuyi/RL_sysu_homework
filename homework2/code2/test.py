import argparse
import numpy as np
import gym
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from wrappers import make_env
seed = 11037
import warnings
warnings.filterwarnings('ignore')

def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    # parser.add_argument('--test_pg', action='store_true', help='whether test policy gradient')
    # parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--test_pg', type=bool, default=False, help='whether test policy gradient')
    parser.add_argument('--test_dqn', type=bool, default=True, help='whether test DQN')

    args = parser.parse_args()
    if args.test_dqn:
        from argument import dqn_arguments
        parser = dqn_arguments(parser)
    elif args.test_pg:
        from argument import pg_arguments
        parser = pg_arguments(parser)
    else:
        raise ValueError('参数错误')
    args = parser.parse_args()
    return args


def test(agent, env, total_episodes=30):
    rewards = []
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        while(not done):
        # while True:
            env.render()
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        print('episode: {}  reward:{:.2f}'.format(i+1, episode_reward))
    env.close()
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_pg:
        env_name = args.env_name
        # env = make_env(env_name)
        env = gym.make(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        test(agent, env, total_episodes=10)

    if args.test_dqn:
        env_name = args.env_name
        env = make_env(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        test(agent, env, total_episodes=10)


if __name__ == '__main__':
    args = parse()
    run(args)
