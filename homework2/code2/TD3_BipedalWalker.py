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
    parser = argparse.ArgumentParser(description="TD3 for BipedalWalker")
    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--device', type=str, default='cpu', help="cpu or cuda")
    parser.add_argument('--lr', default=1e-3, type=float, help='')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--hidden_size', default=256, type=int, help='')
    parser.add_argument('--grad_norm_clip', default=1, type=float, help='')

    parser.add_argument('--num_episodes', default=1000, type=int, help='')
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--buffer_size', default=int(1e6), type=int, help='')
    parser.add_argument('--warm_up', default=int(1e3), type=int, help='')
    parser.add_argument('--update_interval', default=2, type=int, help='')  # TD3延迟更新间隔
    parser.add_argument('--explore_noise', default=0.25, type=float, help='')  # TD3的探索噪声比例
    parser.add_argument('--policy_noise', default=0.2, type=float, help='')  # TD3的策略噪声比例
    parser.add_argument('--noise_clip', default=0.3, type=float, help='')  # 策略噪声的clip
    parser.add_argument('--tau', default=5e-3, type=float, help='')

    parser.add_argument("--exp_name", default='TD3_BipedalWalker', type=str)
    parser.add_argument("--log_dir", default='./result/TD3_BipedalWalker/', type=str)
    parser.add_argument("--save_dir", default='./result/TD3_BipedalWalker/', type=str)  # 学习曲线对应的数据
    parser.add_argument("--checkpoint_dir", default='.result/TD3_BipedalWalker/', type=str)

    args = parser.parse_args()

    return args


class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.size = buffer_size
        self.counter = 0
        self.states = np.zeros((buffer_size, state_dim))
        self.states_new = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))
        self.rewards = np.zeros((buffer_size, 1))
        self.dones = np.zeros((buffer_size, 1), dtype=bool)

    def add(self, state, action, reward, state_new, done):
        index = self.counter % self.size
        self.states[index, :] = state
        self.actions[index, :] = action
        self.rewards[index] = reward
        self.states_new[index, :] = state_new
        self.dones[index] = done
        self.counter = self.counter + 1

    def sample(self, batch_size):
        idx = np.random.choice(np.min([self.size, self.counter]), size=batch_size, replace=False)
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        states_new = self.states_new[idx]
        dones = self.dones[idx]
        return states, actions, rewards, states_new, dones


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128, learning_rate=1e-3, device=t.device('cpu')):
        super(CriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(device)

    def forward(self, state, action):
        s_a = t.cat((state, action), dim=1)
        return self.net(s_a)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_max, hidden_size=128, learning_rate=1e-3, device=t.device('cpu')):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_max = action_max
        self.learning_rate = learning_rate

        self.net = nn.Sequential(
            nn.Linear(self.state_dim , hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),  # nn.ReLU(),
            nn.Linear(hidden_size, self.action_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(device)

    def forward(self, state):
        mu = self.net(state)
        mu = self.action_max * t.tanh(mu)
        return mu


class Agent:
    def __init__(self, env, arglist):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]
        self.action_min = env.action_space.low[0]
        self.warm_up = arglist.warm_up
        self.batch_size = arglist.batch_size
        self.update_interval = arglist.update_interval
        self.gamma = arglist.gamma
        self.tau = arglist.tau  # polyak
        self.explore_noise = arglist.explore_noise
        self.policy_noise = arglist.policy_noise
        self.noise_clip = arglist.noise_clip
        self.grad_norm_clip = arglist.grad_norm_clip
        self.critic_update_counter = 0
        self.play_counter = 0
        self.device = t.device(arglist.device)

        self.replay_buffer = ReplayBuffer(arglist.buffer_size, self.state_dim, self.action_dim)

        params_actor = (self.state_dim, self.action_dim, self.action_max, arglist.hidden_size, arglist.lr, self.device)
        params_critic = (self.state_dim, self.action_dim, arglist.hidden_size, arglist.lr, self.device)
        self.actor = ActorNetwork(*params_actor)
        self.critic_1 = CriticNetwork(*params_critic)
        self.critic_2 = CriticNetwork(*params_critic)

        self.target_actor = ActorNetwork(*params_actor)
        self.target_critic_1 = CriticNetwork(*params_critic)
        self.target_critic_2 = CriticNetwork(*params_critic)

        self.update_target_networks(tau=1)

    def get_action(self, state, training=True):
        state = t.tensor([state], dtype=t.float).to(self.device)
        if self.play_counter <= self.warm_up and training:
            action = self.env.action_space.sample()
        else:
            action = self.actor(state)
            if training:
                action += (self.explore_noise * self.action_max) * t.randn(action.size()).to(self.device)
            action = t.clamp(action, min=self.action_min, max=self.action_max)
            action = action.cpu().detach().numpy()[0]

        self.play_counter += 1
        return action

    def update(self):
        states, actions, rewards, new_states, dones = self.replay_buffer.sample(batch_size=self.batch_size)
        states = t.tensor(states, dtype=t.float).to(self.device)
        actions = t.tensor(actions, dtype=t.float).to(self.device)
        rewards = t.tensor(rewards, dtype=t.float).to(self.device)
        new_states = t.tensor(new_states, dtype=t.float).to(self.device)
        dones = t.tensor(dones).to(self.device)

        target_noise = t.clamp(self.policy_noise*t.randn(actions.size()), min=-self.noise_clip, max=self.noise_clip) * self.action_max
        target_actions = self.target_actor(new_states) + target_noise.to(self.device)
        target_actions = t.clamp(target_actions, min=self.action_min, max=self.action_max)
        target_Q1 = self.target_critic_1(new_states, target_actions)
        target_Q2 = self.target_critic_2(new_states, target_actions)
        target_critic_value = t.min(target_Q1, target_Q2)
        target_critic_value[dones] = 0.0
        target = rewards + self.gamma*target_critic_value

        critic_1_value = self.critic_1(states, actions)
        critic_2_value = self.critic_2(states, actions)

        self.critic_1.optimizer.zero_grad()
        critic_1_loss = F.mse_loss(target, critic_1_value)

        self.critic_2.optimizer.zero_grad()
        critic_2_loss = F.mse_loss(target, critic_2_value)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.grad_norm_clip)
        nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.grad_norm_clip)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.critic_update_counter += 1

        if self.critic_update_counter % self.update_interval == 0:
            self.actor.optimizer.zero_grad()
            actor_loss = self.critic_1(states, self.actor(states))
            actor_loss = -t.mean(actor_loss)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
            self.actor.optimizer.step()

            self.update_target_networks()

    def update_target_networks(self, tau=None):
        if tau is None:
            tau = self.tau

        critic_1_params = dict(self.critic_1.named_parameters())
        target_critic_1_params = dict(self.target_critic_1.named_parameters())
        for name in target_critic_1_params:
            target_critic_1_params[name] = tau*critic_1_params[name].clone() + \
                                           (1 - tau)*target_critic_1_params[name].clone()
        self.target_critic_1.load_state_dict(target_critic_1_params)

        critic_2_params = dict(self.critic_2.named_parameters())
        target_critic_2_params = dict(self.target_critic_2.named_parameters())
        for name in target_critic_2_params:
            target_critic_2_params[name] = tau*critic_2_params[name].clone() + \
                                           (1 - tau)*target_critic_2_params[name].clone()
        self.target_critic_2.load_state_dict(target_critic_2_params)

        actor_params = dict(self.actor.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        for name in target_actor_params:
            target_actor_params[name] = tau*actor_params[name].clone() + \
                                        (1 - tau)*target_actor_params[name].clone()
        self.target_actor.load_state_dict(target_actor_params)


def play_one_game(agent, env):
    total_reward = 0
    observation = env.reset()
    done = False
    cnt = 0
    while not done:
        action = agent.get_action(observation)
        observation_next, reward, done, info = env.step(action)
        total_reward += reward

        agent.replay_buffer.add(observation, action, reward, observation_next, done)
        observation = observation_next
        if agent.replay_buffer.counter > agent.warm_up:
            agent.update()
        # cnt += 1
    # print(cnt)  # 100 - 1000
    return agent, total_reward


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
    env = gym.make('BipedalWalker-v2')  # BipedalWalkerHardcore-v3
    env.seed(arglist.seed)
    agent = Agent(env, arglist)

    if training:
        print('Training...')
        t0 = time.time()
        t_start = t0

        res = []
        avg_res = []
        for i in range(1, arglist.num_episodes+1):
            agent.explore_noise *= 0.999  # 探索噪声递减
            agent, total_reward = play_one_game(agent, env)

            res.append(total_reward)
            avg_res.append(np.mean(res[-100:]))
            if i % 10 == 0:
                print('episode: {}  return: {:.2f}   mean: {:.2f}  time: {:.2f}s'
                      .format(i, total_reward, np.mean(res[-100:]), (time.time() - t0)))
                t0 = time.time()
        print('Time: {:.2f}m'.format((time.time() - t_start) / 60))

        # 保存训练数据
        data = {'res': res, 'avg_res': avg_res}
        file_name = arglist.save_dir + arglist.exp_name + '.pkl'
        with open(file_name, 'wb') as fp:
            pickle.dump(data, fp)

        # # Plotting the results
        # axes = plt.axes()
        # axes.set_ylim([np.min(res) - 5, np.max(res) + 5])
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.plot(np.arange(1, arglist.num_episodes + 1), res)
        # plt.plot(np.arange(1, arglist.num_episodes + 1), avg_res)
        # legend_2 = 'Average of the last 100 episodes'
        # plt.legend(['Reward', legend_2], loc=4)
        # plt.savefig(fname=arglist.save_dir + 'res.png')
        # plt.show()

        # Saving the networks
        t.save(agent.critic_1.state_dict(), arglist.save_dir+'critic_1')
        t.save(agent.critic_2.state_dict(), arglist.save_dir+'critic_2')
        t.save(agent.actor.state_dict(), arglist.save_dir+'actor')
        t.save(agent.target_critic_1.state_dict(), arglist.save_dir+'target_critic_1')
        t.save(agent.target_critic_2.state_dict(), arglist.save_dir+'target_critic_2')
        t.save(agent.target_actor.state_dict(), arglist.save_dir+'target_actor')
    else:
        env.seed(1)
        agent.critic_1.load_state_dict(t.load(arglist.save_dir+'critic_1', map_location='cpu'))
        agent.critic_2.load_state_dict(t.load(arglist.save_dir+'critic_2', map_location='cpu'))
        agent.actor.load_state_dict(t.load(arglist.save_dir+'actor', map_location='cpu'))
        agent.target_critic_1.load_state_dict(t.load(arglist.save_dir+'target_critic_1', map_location='cpu'))
        agent.target_critic_2.load_state_dict(t.load(arglist.save_dir+'target_critic_2', map_location='cpu'))
        agent.target_actor.load_state_dict(t.load(arglist.save_dir+'target_actor', map_location='cpu'))

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
