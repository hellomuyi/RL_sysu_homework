import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import random
import sys
import time
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

def parse():
    parser = argparse.ArgumentParser(description="A2C for CartPole-v0")
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--device', type=str, default='cpu', help="cpu or cuda")
    parser.add_argument('--lr', default=3.4e-4, type=float, help='')  # 3.5e-4
    parser.add_argument('--hidden_size', default=64, type=int, help='')  # 64
    parser.add_argument('--grad_norm_clip', default=10, type=float, help='')  # 9.1

    parser.add_argument('--num_episodes', default=448, type=int, help='')
    parser.add_argument('--gamma', default=0.95, type=float, help='')

    parser.add_argument("--exp_name", default='A2C_CartPole-v0', type=str)
    parser.add_argument("--log_dir", default='./result/A2C/', type=str)
    parser.add_argument("--save_dir", default='./result/A2C/', type=str)  # 学习曲线对应的数据
    parser.add_argument("--checkpoint_dir", default='.result/A2C/', type=str)

    args = parser.parse_args()

    return args


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=128, learning_rate=1e-3, device=t.device('cpu')):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.learning_rate = learning_rate

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(device)

    def forward(self, state):
        return self.net(state)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, learning_rate=1e-3, device=t.device('cpu')):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.net = nn.Sequential(
            nn.Linear(self.state_dim , hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, self.action_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(device)

    def forward(self, state):
        logits = self.net(state)
        return logits


class Agent:
    def __init__(self, env, arglist):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n   # env.action_space.shape[0]
        self.gamma = arglist.gamma
        self.grad_norm_clip = arglist.grad_norm_clip
        self.device = arglist.device

        self.critic = CriticNetwork(state_dim=self.state_dim,
                                    hidden_size=arglist.hidden_size,
                                    learning_rate=arglist.lr,
                                    device=self.device)
        self.actor = ActorNetwork(state_dim=self.state_dim,
                                  action_dim=self.action_dim,
                                  hidden_size=arglist.hidden_size,
                                  learning_rate=arglist.lr,
                                  device=self.device)

    def get_action(self, state, training=True):
        state = t.tensor([state], dtype=t.float).to(self.device)
        logits = self.actor(state)
        p = t.softmax(logits, -1)
        if training:
            # action = self.env.action_space.sample()
            action = t.multinomial(p, 1).item()  # action = action.cpu().detach().numpy()[0]
        else:
            action = t.argmax(p).item()
        return action

    def update(self, obs, action, reward, obs_next, done):
        obs = t.tensor(obs, dtype=t.float).to(self.device)
        action = t.tensor(action, dtype=t.int).to(self.device)
        reward = t.tensor(reward, dtype=t.float).to(self.device)
        obs_next = t.tensor(obs_next, dtype=t.float).to(self.device)
        done = t.tensor(done, dtype=t.float).to(self.device)

        v = self.critic(obs)
        v = (1 - done) * v
        y = reward + self.gamma * self.critic(obs_next)  # 目标critic减缓自举
        delta = (y - v)

        # 更新critic
        critic_loss = nn.MSELoss()(v, y)# delta ** 2  # nn.SmoothL1Loss()(y, v)    # delta ** 2
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm_clip)
        self.critic.optimizer.step()

        # 更新actor
        logits = self.actor(obs)
        log_p = t.log_softmax(logits, dim=-1)
        log_p = log_p[action]
        actor_loss = -log_p * delta.detach()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
        self.actor.optimizer.step()


def play_one_game(agent, env):
    obs = env.reset()
    done = False
    episode_reward = 0.0
    while not done:
        action = agent.get_action(obs, training=True)
        obs_new, reward, done, info = env.step(action)
        agent.update(obs, action, reward, obs_new, done)
        obs = obs_new
        episode_reward += reward
    return agent, episode_reward


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        pass


def main(arglist, training=True):
    set_seed(arglist.seed)
    env = gym.make('CartPole-v0')
    env.seed(arglist.seed)
    # print(env._max_episode_steps)  # 200
    # env._max_episode_steps = 2000

    agent = Agent(env, arglist)
    if training:
        print('Training...')
        t0 = time.time()
        t_start = t0

        res = []
        avg_res = []
        for i in range(arglist.num_episodes):
            agent, total_reward = play_one_game(agent, env)

            res.append(total_reward)
            avg_res.append(np.mean(res[-10:]))
            if i % 10 == 0:
                print('episode: {}  return: {:.2f}   mean: {:.2f}  time: {:.2f}s'
                      .format(i, total_reward, np.mean(res[-10:]), (time.time() - t0)))
                t0 = time.time()
        print('Time: {:.2f}m'.format((time.time() - t_start) / 60))

        # 保存训练数据
        data = {'res': res, 'avg_res': avg_res}
        file_name = arglist.save_dir + arglist.exp_name + '.pkl'
        with open(file_name, 'wb') as fp:
            pickle.dump(data, fp)

        # Plotting the results
        axes = plt.axes()
        axes.set_ylim([np.min(res) - 10, np.max(res) + 10])
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(np.arange(1, arglist.num_episodes + 1), res)
        plt.plot(np.arange(1, arglist.num_episodes + 1), avg_res)
        legend_2 = 'Average of the last 10 episodes'
        plt.legend(['Reward', legend_2], loc=4)
        plt.savefig(fname=arglist.save_dir + arglist.exp_name + '.png')
        plt.show()

        # Saving the networks
        t.save(agent.critic.state_dict(), arglist.save_dir+'critic')
        t.save(agent.actor.state_dict(), arglist.save_dir+'actor')
    else:
        agent.critic.load_state_dict(t.load(arglist.save_dir+'critic'))
        agent.actor.load_state_dict(t.load(arglist.save_dir+'actor'))

        rewards = []
        for i in range(10):
            observation = env.reset()
            done = False

            total_reward = 0
            while not done:
                env.render()
                a = agent.get_action(observation, training=False)
                observation, reward, done, info = env.step(a)
                total_reward = total_reward + reward
            print('test: {:2d}  reward: {:.2f}'.format(i+1, total_reward))
            rewards.append(total_reward)
        env.close()
        print('mean: {:.2f}'.format(np.mean(rewards)))


if __name__ == '__main__':
    arglist = parse()

    log_path = arglist.log_dir + arglist.exp_name + '.log'
    sys.stdout = Logger(log_path, sys.stdout)
    sys.stderr = Logger(log_path, sys.stderr)

    main(arglist, training=False)
