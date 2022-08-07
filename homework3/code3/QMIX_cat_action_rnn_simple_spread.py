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
import copy
import warnings

warnings.filterwarnings('ignore')
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def parse():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

    parser.add_argument("--test", default=True, type=bool)  # 测试时加载已训练好的模型

    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")  # 25
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")  # 60000

    parser.add_argument('--seed', default=1, type=int, help='')
    parser.add_argument('--device', type=str, default='cpu', help="cpu or cuda")
    parser.add_argument('--lr', default=1e-3, type=float, help='')
    parser.add_argument('--batch_size', default=int(256), type=int, help='')  # batch_size是回合数  512/25=20
    parser.add_argument('--hidden_size_drqn', default=64, type=int, help='')
    parser.add_argument('--hidden_size_qmix', default=128, type=int, help='')
    parser.add_argument('--grad_norm_clip', default=0.5, type=float, help='')
    parser.add_argument('--repeats', default=1, type=int, help='')  # 一个回合后更新网络的次数

    parser.add_argument('--gamma', default=0.95, type=float, help='')
    parser.add_argument('--buffer_size', default=int(1e5), type=int, help='')  # buffer_size是episode的数量
    parser.add_argument('--warm_up', default=int(3e2), type=int, help='')
    parser.add_argument('--update_interval', default=100, type=int, help='')  # 每隔n次更新一次目标网络
    parser.add_argument("--eps_begin", default=0.9, type=float)  # epsilon初始值
    parser.add_argument("--eps_end", default=0.01, type=float)   # epsilon终止值

    parser.add_argument("--exp_name", default='QMIX3', type=str)
    parser.add_argument("--log_dir", default='./result/QMIX3/', type=str)
    parser.add_argument("--save_dir", default='./result/QMIX3/', type=str)  # 学习曲线对应的数据
    parser.add_argument("--print_rate", default=100, type=int)  # 每隔save_rate个episodes打印日志
    parser.add_argument("--checkpoint_dir", default='.result/QMIX3/', type=str)

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
    def __init__(self, buffer_size, episode_len, obs_dim, act_dim, num_agents):
        """
        QMIX用到RNN，其隐藏状态的时序性质使得训练必须得以回合为单位
        注意一个episode若有25个step，则共有26个状态
        :param buffer_size: episodes的数量,区别于一般RL的以转移为存放单位,总量为step的数量
        :param episode_len:
        :param obs_dim:
        :param act_dim:
        :param num_agents:
        """
        self.size = buffer_size
        self.counter = 0
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_n_eps_all = np.zeros((buffer_size, episode_len+1, num_agents, obs_dim))  # 存放当时的全局状态(所有智能体的局部观测)
        self.act_n_eps_all = np.zeros((buffer_size, episode_len, num_agents, act_dim))
        self.reward_eps_all = np.zeros((buffer_size, episode_len, 1))
        self.done_eps_all = np.zeros((buffer_size, episode_len, 1), dtype=bool)

    def add(self, obs_n_eps, act_n_eps, reward_eps, done_eps):
        """

        :param obs_n_eps: (episode_len+1, num_agents, obs_dim)
        :param act_n_eps: (episode_len, num_agents, act_dim)
        :param reward_eps:  (episode_len, 1)  同质智能体的奖励是一样的
        :param done_eps:  (episode_len, 1)  simple_spread场景中,所有智能体的done是一致的
        """
        index = self.counter % self.size
        self.obs_n_eps_all[index, :] = obs_n_eps
        self.act_n_eps_all[index, :] = act_n_eps
        self.reward_eps_all[index] = reward_eps
        self.done_eps_all[index] = done_eps
        self.counter = self.counter + 1

    def sample(self, batch_size):
        idx = np.random.choice(np.min([self.size, self.counter]), size=batch_size, replace=False)
        obs_n_eps_bs = self.obs_n_eps_all[idx]  # (bs, episode_len+1, num_agents, obs_dim)
        act_n_eps_bs = self.act_n_eps_all[idx]
        reward_eps_bs = self.reward_eps_all[idx]  # (bs, episode_len, 1)
        done_eps_bs = self.done_eps_all[idx]  # (bs, episode_len, 1)

        return obs_n_eps_bs, act_n_eps_bs, reward_eps_bs, done_eps_bs


class DRQN(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super(DRQN, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(obs_dim+action_dim, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, obs, last_action, hidden_state):
        """

        :param obs: (num_agents, dim_obs) or (bs, num_agents, dim_obs)
        :param last_action:
        :param hidden_state:
        :return:
        """
        x = F.relu(self.fc1(t.cat((obs, last_action), dim=-1)))
        h_in = hidden_state.reshape(-1, self.hidden_size)
        h = self.rnn(x.view(-1, self.hidden_size), h_in)
        q = self.fc2(h)
        # q = self.fc2(x)
        # h = None
        return q, h


class QMIXNet(nn.Module):
    def __init__(self, state_dim, hidden_size, num_agents, is_two_hyper_layers):
        super(QMIXNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_agents = num_agents

        def init(module, weight_init, bias_init, gain=1):
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
            return module

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][0]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        # 输出向量，再reshape为权重矩阵w

        self.hyper_w1 = init_(nn.Linear(state_dim, num_agents))

        self.hyper_b1 = init_(nn.Linear(state_dim, 1))

    def forward(self, q_values, states):
        """
        :param q_values: (bs, num_agents)
        :param states: (bs, obs_n_sim=obs_dim*num_agents)
        :return: (bs, 1)
        """
        q_values = q_values.view(-1, 1, self.num_agents)
        w1 = t.abs(self.hyper_w1(states))  # 权重非负  (bs, num_agents)
        w1 = w1.view(-1, self.num_agents, 1)
        b1 = self.hyper_b1(states)  # (bs, 1)
        b1 = b1.view(-1, 1, 1)

        q_total = t.bmm(q_values, w1) +b1

        q_total = q_total.view(-1, 1)  # (bs, 1)
        return q_total


# class QMIXNet(nn.Module):
#     def __init__(self, obs_n_dim, hidden_size, num_agents, is_two_hyper_layers):
#         super(QMIXNet, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_agents = num_agents
#
#         def init(module, weight_init, bias_init, gain=1):
#             weight_init(module.weight.data, gain=gain)
#             bias_init(module.bias.data)
#             return module
#
#         init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][0]
#         def init_(m):
#             return init(m, init_method, lambda x: nn.init.constant_(x, 0))
#         # 输出向量，再reshape为权重矩阵w
#         if is_two_hyper_layers:
#             self.hyper_w1 = nn.Sequential(init_(nn.Linear(obs_n_dim, hidden_size)),
#                                           nn.ReLU(),
#                                           init_(nn.Linear(hidden_size, num_agents * hidden_size)))
#             self.hyper_w2 = nn.Sequential(init_(nn.Linear(obs_n_dim, hidden_size)),
#                                           nn.ReLU(),
#                                           init_(nn.Linear(hidden_size, hidden_size)))
#         else:
#             self.hyper_w1 = init_(nn.Linear(obs_n_dim, num_agents * hidden_size))
#             self.hyper_w2 = init_(nn.Linear(obs_n_dim, hidden_size))
#
#         self.hyper_b1 = init_(nn.Linear(obs_n_dim, hidden_size))
#         self.hyper_b2 = nn.Sequential(init_(nn.Linear(obs_n_dim, hidden_size)),
#                                       nn.ReLU(),
#                                       init_(nn.Linear(hidden_size, 1)))
#
#     def forward(self, q_values, states):
#         """
#         :param q_values: (bs, num_agents)
#         :param states: (bs, obs_n_sim=obs_dim*num_agents)
#         :return: (bs, 1)
#         """
#         q_values = q_values.view(-1, 1, self.num_agents)
#         w1 = t.abs(self.hyper_w1(states))  # 权重非负  (bs, num_agents*hidden_size)
#         b1 = self.hyper_b1(states)  # (bs, hidden_size)
#
#         w1 = w1.view(-1, self.num_agents, self.hidden_size)  # (bs, num_agents, hidden_size)
#         b1 = b1.view(-1, 1, self.hidden_size)  # (bs, 1, hidden_size)
#
#         hidden = F.elu(t.bmm(q_values, w1) + b1)  # (bs, 1, hidden_size)
#
#         # 第二层
#         w2 = t.abs(self.hyper_w2(states))  # 权重非负  (bs, hidden_size)
#         b2 = self.hyper_b2(states)  # (bs, 1)
#
#         w2 = w2.view(-1, self.hidden_size, 1)  # (bs, hidden_size, 1)
#         b2 = b2.view(-1, 1, 1)  # (bs, 1, 1)
#
#         q_total = t.bmm(hidden, w2) + b2  # (bs, 1, 1)
#         q_total = q_total.view(-1, 1)  # (bs, 1)
#         return q_total


class QMIXAgent:
    def __init__(self, env, arglist):
        self.args = arglist
        self.test = arglist.test
        self.env = env
        self.num_agents = env.n
        self.obs_dim = env.observation_space[0].shape[0]
        self.act_dim = env.action_space[0].n
        self.max_episode_len = arglist.max_episode_len
        self.hidden_size_drqn = arglist.hidden_size_drqn
        self.hidden_size_qmix = arglist.hidden_size_qmix
        self.warm_up = arglist.warm_up
        self.batch_size = arglist.batch_size
        self.gamma = arglist.gamma
        self.grad_norm_clip = arglist.grad_norm_clip
        self.learning_step = 0
        self.play_counter = 0
        self.repeats = arglist.repeats
        self.update_interval = arglist.update_interval
        self.device = t.device(arglist.device)
        self.eps_begin, self.eps_end = arglist.eps_begin, arglist.eps_end
        self.epsilon = self.eps_begin
        self.epsilons = self.get_epsilons()


        self.replay_buffer = ReplayBuffer(arglist.buffer_size, self.max_episode_len, self.obs_dim, self.act_dim, env.n)

        params_drqn = (self.obs_dim, self.act_dim, arglist.hidden_size_drqn)
        params_qmix = ((self.hidden_size_drqn+self.act_dim+self.obs_dim) * self.num_agents, arglist.hidden_size_qmix, self.num_agents, False)
        self.eval_drqn_net = DRQN(*params_drqn).to(self.device)
        self.target_drqn_net = DRQN(*params_drqn).to(self.device)

        self.eval_qmix_net = QMIXNet(*params_qmix).to(self.device)
        self.target_qmix_net = QMIXNet(*params_qmix).to(self.device)

        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_drqn_net.parameters())
        self.optim = optim.Adam(self.eval_parameters, lr=arglist.lr)

    def get_epsilons(self):
        # 一次性计算出各episode对应的epsilon
        power = np.array(5.0)
        max_p, min_p = self.eps_begin, self.eps_end
        num_episodes = self.args.num_episodes
        num_episodes_decay = num_episodes // 5  # 200
        dif = (np.power(max_p, 1 / power) - np.power(min_p, 1 / power)) / (num_episodes_decay)
        epsilons = np.ones(num_episodes, dtype=np.float64) * min_p
        epsilons[0] = max_p
        for i in range(1, num_episodes_decay):
            epsilons[i] = np.power(np.power(epsilons[i - 1], (1 / power)) - dif, power)
        return epsilons

    def get_action(self, obs, last_action, eval_hidden):
        """
        :param obs: (num_agents, dim_act)
        :param last_action:
        :param training:
        :return: (num_action, dim_act)
        """
        obs = t.tensor(obs, dtype=t.float).to(self.device)
        last_action = t.tensor(last_action, dtype=t.int64).to(self.device)

        logits, eval_hidden = self.eval_drqn_net(obs, last_action, eval_hidden)
        action = np.zeros((self.num_agents, self.act_dim), dtype=np.int64)
        if not self.test:  # 训练，epsilon贪心
            if np.random.rand() < self.epsilon:
                idx = np.random.randint(0, self.act_dim, (self.num_agents,))
                action[np.arange(self.num_agents), idx] = 1
            else:
                idx = t.argmax(logits, -1).cpu().numpy()
                action[np.arange(self.num_agents), idx] = 1
        else:  # 测试，不采用epsilon贪心
            idx = t.argmax(logits, -1).cpu().numpy()
            action[np.arange(self.num_agents), idx] = 1
        return action, eval_hidden

    def update(self):
        obs_n_eps, act_n_eps, reward_n_eps, done_n_eps = self.replay_buffer.sample(batch_size=self.batch_size)
        obs_n_eps = t.tensor(obs_n_eps, dtype=t.float).to(self.device)  # (bs, episode_len+1, num_agents, obs_dim)
        act_n_eps = t.tensor(act_n_eps, dtype=t.int32).to(self.device)
        reward_n_eps = t.tensor(reward_n_eps, dtype=t.float).to(self.device)  # (bs, episode_len, 1)
        done_n_eps = t.tensor(done_n_eps).to(self.device)  # (bs, episode_len, 1)

        # 初始化上一时刻的动作 及 RNN的隐藏状态
        eval_hidden = t.zeros((self.batch_size, self.num_agents, self.hidden_size_drqn)).to(self.device)
        target_hidden = t.zeros((self.batch_size, self.num_agents, self.hidden_size_drqn)).to(self.device)
        last_act_n = t.zeros((self.batch_size, self.num_agents, self.act_dim), dtype=t.int64).to(self.device)
        # QMIX用到RNN，其隐藏状态的时序性质使得训练必须得以episode为单位
        losses = []
        for idx_step in range(self.max_episode_len):
            obs_n = obs_n_eps[:, idx_step, :, :]  # (bs, num_agents, obs_dim)
            act_n = act_n_eps[:, idx_step, :, :]  # (bs, num_agents, act_dim)
            new_obs_n = obs_n_eps[:, idx_step+1, :, :]  # (bs, num_agents, obs_dim)
            reward_n = reward_n_eps[:, idx_step, :]  # (bs, 1)
            done_n = done_n_eps[:, idx_step, :]  # (bs, 1)

            # 计算Q值
            eval_hidden_ = eval_hidden.clone().detach()
            q_eval, eval_hidden = self.eval_drqn_net(obs_n, last_act_n, eval_hidden)  # 输出(bs, num_agents, num_act) (bs, num_agents, hidden_size_drqn)
            q_eval = q_eval.view(self.batch_size, self.num_agents, self.act_dim)
            # eval_hidden = eval_hidden.data
            act_n_ = t.max(act_n, -1)[1]
            q_eval = q_eval.gather(2, act_n_.unsqueeze(2)).squeeze(2)  # 动作是整数  输出(bs, num_agents)
            # q_eval = t.sum(q_eval * act_n, -1)  # 动作是one-hot向量   输出(bs, num_agents)

            # 拼接所有agent局部观测和上一时刻的动作, 作为全局状态
            states = t.cat((obs_n, last_act_n, eval_hidden_.view(self.batch_size, self.num_agents, self.hidden_size_drqn)), dim=-1)
            q_total_eval = self.eval_qmix_net(q_eval, 
                                              states.view(self.batch_size, (self.obs_dim+self.act_dim+self.hidden_size_drqn)*self.num_agents))
            # 计算目标Q值
            target_hidden_ = target_hidden.clone().detach()
            q_next, target_hidden = self.target_drqn_net(new_obs_n, act_n, target_hidden)  # 输出(bs, num_agents, num_act) (bs, num_agents, hidden_size_drqn)
            q_next = q_next.view(self.batch_size, self.num_agents, self.act_dim)

            # double DQN: 下一时刻的动作为q_eval最大的动作
            q_eval_next, _ = self.eval_drqn_net(new_obs_n, act_n, target_hidden_)
            q_eval_next = q_eval_next.view(self.batch_size, self.num_agents, self.act_dim)  # (bs, num_agents, num_act)
            act_next = t.max(q_eval_next, dim=-1)[1]  # (bs, num_agents)
            act_next = act_next.view(-1)
            q_next = q_next.view(self.batch_size*self.num_agents, self.act_dim)[t.arange(self.batch_size*self.num_agents), act_next]
            q_next = q_next.view(self.batch_size, self.num_agents)
            # q_next = t.max(q_next, -1)[0]  # (bs, num_agents)
            q_next = q_next.detach()

            # 基于值分解,计算目标团队Q值
            states_next = t.cat((new_obs_n, act_n, target_hidden_.view(self.batch_size, self.num_agents, self.hidden_size_drqn)), dim=-1)
            q_total_next = self.target_qmix_net(q_next,
                                                states_next.view(self.batch_size,
                                                            (self.obs_dim + self.act_dim+self.hidden_size_drqn) * self.num_agents))
            # q_total_next = t.sum(q_next, -1, keepdim=True)
            q_total_next[done_n] = 0.
            targets = reward_n + self.gamma * q_total_next.detach()
            losses.append(t.mean((q_total_eval - targets) ** 2))
            # losses.append(nn.SmoothL1Loss()(q_total_eval, targets))

            last_act_n = act_n  # last_act来自经验池还是eval_net，来自经验池

        self.optim.zero_grad()
        loss = t.mean(t.stack(losses)) / self.max_episode_len
        # if self.learning_step % 10 == 0:
        #     print(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_parameters, self.grad_norm_clip)
        self.optim.step()


        if self.learning_step % self.update_interval == 0:
            self.target_drqn_net.load_state_dict(self.eval_drqn_net.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        self.learning_step += 1


def display(arglist):
    env = make_env(arglist.scenario, arglist)
    env.seed(arglist.seed)
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
        print('test: {:2d}  reward: {:.2f}'.format(i + 1, total_reward))
        rewards.append(total_reward)
    env.close()
    print('mean: {:.2f}'.format(np.mean(rewards)))


def play_one_game(agent, env):
    # 初始化eval_hidden
    eval_hidden_n = t.zeros((agent.num_agents, agent.hidden_size_drqn)).to(agent.device)
    # 初始化lastaction
    last_act_n = t.zeros((agent.num_agents, agent.act_dim), dtype=t.int64).to(agent.device)

    total_reward = 0
    obs_n = env.reset()  # list
    obs_n_eps = np.zeros((agent.max_episode_len+1, agent.num_agents, agent.obs_dim))
    act_n_eps = np.zeros((agent.max_episode_len, agent.num_agents, agent.act_dim), dtype=np.int64)
    reward_eps = np.zeros((agent.max_episode_len, 1))
    done_eps = np.zeros((agent.max_episode_len, 1), dtype=np.bool)

    obs_n_eps[0] = obs_n
    for i in range(arglist.max_episode_len):
        if agent.test:
            time.sleep(0.1)
            env.render(mode='no')
        # act_n = [agent.get_action(obs_n[i]) for i in range(env.n)]
        act_n, eval_hidden_n = agent.get_action(np.array(obs_n), last_act_n, eval_hidden_n)
        act_n_eps[i] = act_n
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(act_n)  # 都是list
        done = all(done_n)
        obs_n_eps[i+1] = new_obs_n
        reward_eps[i] = rew_n[0]
        done_eps[i] = done

        total_reward += rew_n[0]  # 合作型场景中，每个智能体的奖励是相同的，取第一个奖励

        # if agent.replay_buffer.counter > agent.warm_up:
        #     agent.update()

        if done:
            break
        obs_n = new_obs_n
        last_act_n = act_n
    agent.replay_buffer.add(obs_n_eps, act_n_eps, reward_eps, done_eps)

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


def main(arglist):
    set_seed(arglist.seed)
    env = make_env(arglist.scenario, arglist)
    # env.seed(arglist.seed)
    agent = QMIXAgent(env, arglist)
    if not arglist.test:
        print('Training...')
        t0 = time.time()
        t_start = t0

        res = []
        avg_res = []
        for i in range(1, arglist.num_episodes + 1):
            agent.epsilon = agent.epsilons[i-1]
            agent, total_reward = play_one_game(agent, env)

            total_reward /= arglist.max_episode_len
            res.append(total_reward)
            avg_res.append(np.mean(res[-100:]))
            if i % arglist.print_rate == 0:
                print('episode: {}  return: {:.2f}   mean: {:.2f}  epsilon: {:.4f}  time: {:.2f}s'
                      .format(i, total_reward, np.mean(res[-100:]), agent.epsilon, (time.time() - t0)))
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
        t.save(agent.eval_drqn_net.state_dict(), arglist.save_dir + 'eval_drqn')
    else:
        # env.seed(1)
        agent.eval_drqn_net.load_state_dict(t.load(arglist.save_dir + 'eval_drqn', map_location='cpu'))

        rewards = []
        for i in range(10):
            agent, total_reward = play_one_game(agent, env)
            total_reward /= arglist.max_episode_len
            print('test: {:2d}  reward: {:.2f}'.format(i + 1, total_reward))
            rewards.append(total_reward)
        env.close()
        print('mean: {:.2f}'.format(np.mean(rewards)))


if __name__ == '__main__':
    arglist = parse()

    log_path = arglist.log_dir + arglist.exp_name + '.log'
    sys.stdout = Logger(log_path, sys.stdout)
    sys.stderr = Logger(log_path, sys.stderr)
    main(arglist)
    # display(arglist)
