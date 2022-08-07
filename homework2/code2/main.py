import os, sys
import argparse
import torch, random
import numpy as np
import gym

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(__file__))
from wrappers import make_env
from argument import dqn_arguments, pg_arguments


def parse():
    parser = argparse.ArgumentParser(description="SYSU_RL_HW2")
    parser.add_argument('--train_pg', default=False, type=bool, help='whether train policy gradient')
    parser.add_argument('--train_dqn', default=True, type=bool, help='whether train DQN')

    args = parser.parse_args()
    if args.train_dqn:
        parser = dqn_arguments(parser)
    elif args.train_pg:
        parser = pg_arguments(parser)
    else:
        raise ValueError('参数错误')
    args = parser.parse_args()

    return args


def run(args):
    if args.train_pg:
        env_name = args.env_name
        env = gym.make(env_name)
        # env = make_env(env_name)
        from agent_dir.agent_pg import AgentPG
        agent = AgentPG(env, args)
        agent.run()

    if args.train_dqn:
        env_name = args.env_name
        env = make_env(env_name)
        # env = gym.make(env_name)
        from agent_dir.agent_dqn import AgentDQN
        agent = AgentDQN(env, args)
        agent.run()


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()  # 不启动缓冲
        self.log.flush()

    def flush(self):
        pass


if __name__ == '__main__':
    args = parse()
    set_seed(args.seed)
    log_path = args.log_dir + args.exp_name + '.log'
    sys.stdout = Logger(log_path, sys.stdout)
    sys.stderr = Logger(log_path, sys.stderr)
    run(args)
