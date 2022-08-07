import numpy as np
import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import sys
import time
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def parse():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")  # 25
    parser.add_argument("--num-episodes", type=int, default=50000, help="number of episodes")  # 60000

    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--device', type=str, default='cpu', help="cpu or cuda")
    parser.add_argument('--lr', default=1e-3, type=float, help='')
    parser.add_argument('--batch_size', default=1024, type=int, help='')
    parser.add_argument('--hidden_size', default=128, type=int, help='')
    parser.add_argument('--grad_norm_clip', default=0.5, type=float, help='')  # 0.2
    parser.add_argument('--repeats', default=1, type=int, help='')  # 一个回合后更新网络的次数

    parser.add_argument('--gamma', default=0.95, type=float, help='')
    parser.add_argument('--buffer_size', default=int(1e6), type=int, help='')
    parser.add_argument('--warm_up', default=int(2e3), type=int, help='')
    parser.add_argument('--update_interval', default=2, type=int, help='')  # TD3延迟更新间隔
    parser.add_argument('--explore_noise', default=0.01, type=float, help='')  # TD3的探索噪声比例
    parser.add_argument('--policy_noise', default=0.01, type=float, help='')  # TD3的策略噪声比例
    parser.add_argument('--noise_clip', default=0.01, type=float, help='')  # 策略噪声的clip
    parser.add_argument('--noise_decay', default=0.99, type=float, help='')  # 探索噪声衰减因子
    parser.add_argument('--tau', default=1e-2, type=float, help='')

    parser.add_argument("--exp_name", default='MATD3', type=str)
    parser.add_argument("--log_dir", default='./result/MATD3/', type=str)
    parser.add_argument("--save_dir", default='./result/MATD3/', type=str)  # 学习曲线对应的数据
    parser.add_argument("--print_rate", default=100, type=int)  # 每隔save_rate个episodes打印日志
    parser.add_argument("--checkpoint_dir", default='.result/MATD3/', type=str)

    args = parser.parse_args()

    return args


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()  # 会指定所有智能体的数量
    # create multiagent environment
    #                                                                 观测函数(agent)
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


class ReplayBuffer:
    def __init__(self, buffer_size, obs_dim, act_dim, num_agents):
        self.size = buffer_size
        self.counter = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_n_all = np.zeros((buffer_size, obs_dim*num_agents))  # 存放当时的全局状态
        self.new_obs_n_all = np.zeros((buffer_size, obs_dim*num_agents))
        self.act_n_all = np.zeros((buffer_size, act_dim*num_agents))
        self.reward_all = np.zeros((buffer_size, 1))
        self.done_all = np.zeros((buffer_size, 1), dtype=bool)

    def add(self, obs_n, act_n, reward, new_obs_n, done):
        index = self.counter % self.size
        self.obs_n_all[index, :] = obs_n
        self.act_n_all[index, :] = act_n
        self.reward_all[index] = reward
        self.new_obs_n_all[index, :] = new_obs_n
        self.done_all[index] = done
        self.counter = self.counter + 1

    def sample(self, batch_size):
        idx = np.random.choice(np.min([self.size, self.counter]), size=batch_size, replace=False)
        obs_n_bs = self.obs_n_all[idx]             # (bs, obs_dim*3)
        act_n_bs = self.act_n_all[idx]             # (bs, act_dim*3)
        reward_bs = self.reward_all[idx]
        new_obs_n_bs = self.new_obs_n_all[idx]     # # (bs, obs_dim*3)
        done_bs = self.done_all[idx]

        # obs_bs = obs_n_bs[:, :self.obs_dim]
        # act_bs = act_n_bs[:, :self.act_dim]
        # new_obs_bs = new_obs_n_bs[:, :self.obs_dim]

        return obs_n_bs, act_n_bs, reward_bs, new_obs_n_bs, done_bs


class CriticNetwork(nn.Module):
    def __init__(self, obs_n_dim, act_n_dim, hidden_size=128, learning_rate=1e-3, device=t.device('cpu')):
        super(CriticNetwork, self).__init__()
        self.obs_n_dim = obs_n_dim
        self.act_n_dim = act_n_dim
        self.learning_rate = learning_rate

        self.net = nn.Sequential(
            nn.Linear(self.obs_n_dim + self.act_n_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(device)

    def forward(self, obs_n, act_n):
        s_a = t.cat((obs_n, act_n), dim=1)
        return self.net(s_a)


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128, learning_rate=1e-3, device=t.device('cpu')):
        super(ActorNetwork, self).__init__()
        self.state_dim = obs_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.net = nn.Sequential(
            nn.Linear(self.state_dim , hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.to(device)

    def forward(self, obs):
        mu = self.net(obs)
        # mu = self.action_max * t.tanh(mu)
        return mu


class MADDPGAgent:
    def __init__(self, env, arglist):
        self.env = env
        self.num_agents = env.n
        self.obs_dim = env.observation_space[0].shape[0]
        self.act_dim = env.action_space[0].n
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
        self.repeats = arglist.repeats
        self.device = t.device(arglist.device)

        self.replay_buffer = ReplayBuffer(arglist.buffer_size, self.obs_dim, self.act_dim, env.n)

        params_actor = (self.obs_dim, self.act_dim, arglist.hidden_size, arglist.lr, self.device)
        params_critic = (self.obs_dim*self.num_agents, self.act_dim*self.num_agents, arglist.hidden_size, arglist.lr, self.device)
        self.actor = ActorNetwork(*params_actor)
        self.critic_1 = CriticNetwork(*params_critic)
        self.critic_2 = CriticNetwork(*params_critic)

        self.target_actor = ActorNetwork(*params_actor)
        self.target_critic_1 = CriticNetwork(*params_critic)
        self.target_critic_2 = CriticNetwork(*params_critic)

        self.update_target_networks(tau=1)

    def gumbel_softmax_action(self, logits):
        """
        gumbel-softmax重参数技巧
        :param logits:
        """
        u = t.rand(logits.shape).to(self.device)
        tau = 0.5  # 温度系数
        return t.softmax((logits - t.log(-t.log(u))) / tau, dim=-1)
        # return t.softmax(logits, dim=-1)

    def get_action(self, obs, training=True):
        obs = t.tensor([obs], dtype=t.float).to(self.device)
        if self.play_counter <= self.warm_up and training:
            action = np.zeros(self.act_dim)
            action[np.random.randint(0, self.act_dim)] = 1
        else:
            logits = self.actor(obs)
            # # gumbel softmax重采样
            action = self.gumbel_softmax_action(logits)
            if training:  # 探索噪声
                action += self.explore_noise * t.randn(action.size()).to(self.device)
            action = t.clamp(action, min=0, max=1)
            action = action.cpu().detach().numpy()[0]

        self.play_counter += 1
        return action

    def update(self):
        obs_n, act_n, reward_n, new_obs_n, done_n = self.replay_buffer.sample(batch_size=self.batch_size)
        obs_n = t.tensor(obs_n, dtype=t.float).to(self.device)
        act_n = t.tensor(act_n, dtype=t.float).to(self.device)
        reward_n = t.tensor(reward_n, dtype=t.float).to(self.device)
        new_obs_n = t.tensor(new_obs_n, dtype=t.float).to(self.device)
        done_n = t.tensor(done_n).to(self.device)

        # 计算下一时刻各智能体的动作a'
        next_act_n = []
        for i in range(self.num_agents):
            act = self.gumbel_softmax_action(self.target_actor(new_obs_n[:, i*self.obs_dim:(i+1)*self.obs_dim]))
            target_noise = t.clamp(self.policy_noise * t.randn(act.size()), min=-self.noise_clip, max=self.noise_clip).to(self.device)
            act = act + target_noise
            act = t.clamp(act, min=0, max=1)
            next_act_n.append(act)
        next_act_n = t.cat(next_act_n, dim=-1)  # (bs, act_dim*n)
        target_Q1 = self.target_critic_1(new_obs_n, next_act_n)
        target_Q2 = self.target_critic_2(new_obs_n, next_act_n)
        target_critic_value = t.min(target_Q1, target_Q2)
        target_critic_value[done_n] = 0.0
        target = reward_n + self.gamma*target_critic_value

        critic_1_value = self.critic_1(obs_n, act_n)  # 动作是当前局部智能体的动作
        critic_2_value = self.critic_2(obs_n, act_n)

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
            logits = self.actor(obs_n[:, :self.obs_dim])
            act_cur = self.gumbel_softmax_action(logits)
            act_n[:, :self.act_dim] = act_cur
            self.actor.optimizer.zero_grad()
            pg_loss = self.critic_1(obs_n, act_n)
            pg_loss = -t.mean(pg_loss)
            p_reg = t.mean(t.square(logits))  # 熵正则化
            # p_reg = 0.  # 实验不使用熵正则化
            actor_loss = pg_loss + p_reg * 1e-3
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


def display(arglist):
    env = make_env(arglist.scenario, arglist)
    rewards = []
    for i in range(10):
        total_reward = 0
        for _ in range(arglist.max_episode_len):
            env.render()
            actions = []
            for _ in range(env.n):
                a = np.zeros(env.action_space[0].n)
                a[np.random.randint(0, 5)] = 1
                actions.append(a)
            obs_n, rew_n, done_n, info_n = env.step(actions)
            total_reward = total_reward + rew_n[0]
            if all(done_n):
                break
        print('test: {:2d}  reward: {:.2f}'.format(i+1, total_reward))
        rewards.append(total_reward)
    env.close()
    print('mean: {:.2f}'.format(np.mean(rewards)))


def play_one_game(agent, env):
    total_reward = 0
    obs_n = env.reset()
    cnt = 0
    for _ in range(arglist.max_episode_len):
        act_n = [agent.get_action(obs_n[i]) for i in range(env.n)]  # 可以不用列表解析:转成而维张量(num_agents, num_action)

        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(act_n)  # 都是list
        done = all(done_n)

        total_reward += rew_n[0]  # 合作型场景中，每个智能体的奖励是相同的，取第一个奖励

        orders = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
        for order in orders:
            agent.replay_buffer.add(np.concatenate(np.array(obs_n)[order]),
                                    np.concatenate(np.array(act_n)[order]),
                                    rew_n[0],
                                    np.concatenate(np.array(new_obs_n)[order]),
                                    done)  # 同质智能体的done是一致的

        # if agent.replay_buffer.counter > agent.warm_up:
        #     agent.update()

        if done:
            break
        obs_n = new_obs_n

    if agent.replay_buffer.counter > agent.warm_up:
        for _ in range(agent.repeats):
            agent.update()
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
    # env = gym.make('BipedalWalker-v2')  # BipedalWalkerHardcore-v3
    env = make_env(arglist.scenario, arglist)
    agent = MADDPGAgent(env, arglist)
    if training:
        print('Training...')
        t0 = time.time()
        t_start = t0

        res = []
        avg_res = []
        for i in range(1, arglist.num_episodes+1):
            agent.explore_noise *= arglist.noise_decay  # 探索噪声递减
            agent, total_reward = play_one_game(agent, env)

            total_reward /= arglist.max_episode_len
            res.append(total_reward)
            avg_res.append(np.mean(res[-100:]))
            if i % arglist.print_rate == 0:
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
        set_seed(2)
        agent.critic_1.load_state_dict(t.load(arglist.save_dir+'critic_1', map_location='cpu'))
        agent.critic_2.load_state_dict(t.load(arglist.save_dir+'critic_2', map_location='cpu'))
        agent.actor.load_state_dict(t.load(arglist.save_dir+'actor', map_location='cpu'))
        agent.target_critic_1.load_state_dict(t.load(arglist.save_dir+'target_critic_1', map_location='cpu'))
        agent.target_critic_2.load_state_dict(t.load(arglist.save_dir+'target_critic_2', map_location='cpu'))
        agent.target_actor.load_state_dict(t.load(arglist.save_dir+'target_actor', map_location='cpu'))

        rewards = []
        for i in range(10):
            total_reward = 0
            obs_n = env.reset()
            for _ in range(arglist.max_episode_len):
                time.sleep(0.1)
                env.render(mode='no')
                act_n = [agent.get_action(obs_n[i], training=False) for i in range(env.n)]
                obs_n, rew_n, done_n, info_n = env.step(act_n)
                total_reward = total_reward + rew_n[0] / arglist.max_episode_len
                if all(done_n):
                    break
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
    # display(arglist)

