import os
import random
import copy
import time
import pickle
import math

import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        # 三层卷积层+双层全连接网络
        self.conv = nn.Sequential(
            nn.Conv2d(input_size[0],
                      out_channels=32,
                      kernel_size=8,
                      stride=4),   # 第一层卷积  (N, 8, 52, 39)
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),  # 第二层卷积  (N, 8, 25, 19)
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),  # 第二层卷积  (N, 8, 25, 19)
            nn.ReLU(),
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(7*7*64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        ##################

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        # 与环境交互式，Q的输入是(H,W,C)
        # 以batch数据训练时，Q的输入是(N,H,W,C)
        if inputs.ndim == 3:
            inputs = np.expand_dims(inputs, axis=0)
        inputs = torch.from_numpy(inputs).to(torch.float32)  # (bs, 210,160,3)
        x = self.conv(inputs)
        x = x.reshape(x.shape[0], -1)  # 展开为向量
        x = self.feed_forward(x)
        return x
        ##################


class DuelingQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DuelingQNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        # 三层卷积层+双层全连接网络
        self.conv = nn.Sequential(
            nn.Conv2d(input_size[0],
                      out_channels=32,
                      kernel_size=8,
                      stride=4),   # 第一层卷积  (N, 8, 52, 39)
            nn.ReLU(),
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=4,
                      stride=2),  # 第二层卷积  (N, 8, 25, 19)
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1),  # 第二层卷积  (N, 8, 25, 19)
            nn.ReLU(),
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(7*7*64, hidden_size),
            nn.ReLU()
        )
        self.V = nn.Linear(hidden_size, 1)
        self.A = nn.Linear(hidden_size, output_size)

        ##################

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        if inputs.ndim == 3:
            inputs = np.expand_dims(inputs, axis=0)
        inputs = torch.from_numpy(inputs).to(torch.float32)  # (bs, 4, 84, 84)
        x = self.conv(inputs)
        x = x.reshape(x.shape[0], -1)  # 展开为向量
        x = self.feed_forward(x)
        V = self.V(x)
        A = self.A(x)
        Q_value = V + (A - torch.mean(A, dim=1, keepdim=True))
        return Q_value
        ##################


class ReplayBuffer:
    def __init__(self, buffer_size):
        ##################
        # YOUR CODE HERE #
        # Max number of transitions to store in the buffer. When the buffer
        #             overflows the old memories are dropped.
        self._storage = []  # 存放五元组，最大长度_maxsize
        self._maxsize = int(buffer_size)
        self._next_idx = 0
        ##################
        pass

    def __len__(self):
        ##################
        # YOUR CODE HERE #
        return len(self._storage)
        ##################
        pass

    def push(self, *transition):
        ##################
        # YOUR CODE HERE #
        data = transition    # (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):  # 未满时尾插
            self._storage.append(data)
        else:  # 满时替换
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        ##################
        pass

    def _encode_sample(self, idxes):
        # 对一个batch的五元组，进行拆分并再次拼接。
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        ##################
        # YOUR CODE HERE #
        # 随机采样
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        # # 最新样本采样
        # idxes = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        # np.random.shuffle(idxes)
        return self._encode_sample(idxes)
        ##################
        pass

    def clean(self):
        ##################
        # YOUR CODE HERE #
        self._storage = []
        self._next_idx = 0
        ##################
        pass


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0
    full = False

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0
            self.full = True

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class PriorizedReplayBuffer(object):
    """Qloss需要乘重要性采样权重
    is weight作为Qloss的权重"""
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.6/(200 * 1000)    # episodes*repeats
    abs_err_upper = 5.  # 1 clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(int(capacity))

    def push(self, *transition):    # transition为一个记录，而不是批
        """
        新加入的经验不用计算TD error，始终以最大优先级存入，保证至少使用一次。
        批训练时，计算TD error，再更新 sum tree
        :param obs_t:
        :param action:
        :param reward:
        :param obs_tp1:
        :param done:
        """
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        # b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1  缓慢增长至1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            # if math.isnan(a) or math.isnan(b):
            #     print(self.tree.tree)
            #     print('*'*60)
            #     print(self.tree.data)
            #     print('*' * 60)
            #     print('min_prob={}, totalP={}'.format(min_prob, self.tree.total_p))
            #     print(pri_seg, i, n)
            #     print(a, b)
            #
            #     import pickle
            #     with open('./debug_tree.pwk', 'wb') as fp:
            #         pickle.dump(self.tree.tree, fp)
            #     with open('./debug_data.pwk', 'wb') as fp:
            #         pickle.dump(self.tree.data, fp)
            #     raise ValueError('a=nan or b=nan')
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)

            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            # b_idx[i], b_memory[i, :] = idx, data
            b_idx[i] = idx
        return b_idx, np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)         # upper?
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def isfull(self):
        return self.tree.full


class AgentDQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        self.args = args
        self.env = env
        self.epsilon = self.args.eps_begin
        self.epsilons = self.get_epsilons()
        self.replay_buffer = PriorizedReplayBuffer(args.buffer_size)

        self.learning_step = 0
        o_dim = env.observation_space.shape  # （210,160,3） --> (4, 84, 84)
        self.action_dim = env.action_space.n  # 6
        self.eval_net = QNetwork(o_dim, args.hidden_size, self.action_dim)
        self.target_net = QNetwork(o_dim, args.hidden_size, self.action_dim)
        # self.eval_net = DuelingQNetwork(o_dim, args.hidden_size, self.action_dim)
        # self.target_net = DuelingQNetwork(o_dim, args.hidden_size, self.action_dim)
        if self.args.test:
            model_path = self.args.model_dir + self.args.exp_name + '.pt'
            self.eval_net.load_state_dict(torch.load(model_path))
        # self.loss_fn = nn.MSELoss()
        self.optim = optim.Adam(self.eval_net.parameters(), lr=args.lr)
        self.optim = optim.RMSprop(self.eval_net.parameters(), lr=args.lr, eps=0.001, alpha=0.95)

        ##################
    
    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass

    def get_epsilons(self):
        # 一次性计算出各episode对应的epsilon
        power = np.array(5.0)
        max_p, min_p = self.args.eps_begin, self.args.eps_end
        num_episodes = self.args.num_episodes
        num_episodes_decay = 100
        dif = (np.power(max_p, 1 / power) - np.power(min_p, 1 / power)) / (num_episodes_decay)
        epsilons = np.ones(num_episodes, dtype=np.float64) * min_p
        epsilons[0] = max_p
        for i in range(1, num_episodes_decay):
            epsilons[i] = np.power(np.power(epsilons[i - 1], (1 / power)) - dif, power)
        return epsilons

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        # # 从经验池采样
        # if self.epsilon > self.args.eps_end:
        #     self.epsilon *= self.args.eps_decay

        if self.learning_step % self.args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learning_step += 1

        # obs, act, rew, obs_next, done = self.replay_buffer.sample(self.args.batch_size)
        tree_idx, obs, act, rew, obs_next, done, ISWeights = self.replay_buffer.sample(self.args.batch_size)
        act = torch.LongTensor(act)
        rew = torch.FloatTensor(rew)
        done = torch.IntTensor(done)
        ISWeights = torch.FloatTensor(ISWeights)
        q_eval = self.eval_net(obs).gather(1, act.unsqueeze(1)).squeeze(1)
        q_next = self.target_net(obs_next).detach()
        # # 1. Nature DQN
        # q_target = rew + self.args.gamma * (1-done) * torch.max(q_next, dim=-1)[0]

        # 2. double DQN  dueling DQN
        act_next = torch.max(self.eval_net(obs_next), dim=-1)[1]
        q_target = rew + self.args.gamma * (1-done) * q_next[torch.arange(len(rew)), act_next]

        # loss = self.loss_fn(q_eval, q_target)
        abs_error = torch.abs(q_eval - q_target)
        # loss = torch.mean(abs_error * ISWeights.squeeze(-1))
        # loss = torch.mean((q_eval - q_target)**2 * ISWeights.squeeze(-1))
        # loss = torch.mean((q_eval - q_target) ** 2)
        loss = nn.SmoothL1Loss()(q_eval*ISWeights.squeeze(-1), q_target*ISWeights.squeeze(-1))
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), self.args.grad_norm_clip)
        self.optim.step()

        if self.learning_step % 2000 == 0:
            print('abs_error: {}, {}, {}, {}'.format(torch.min(abs_error),
                                                     torch.mean(abs_error),
                                                     torch.max(abs_error),
                                                     torch.sum(abs_error > torch.mean(abs_error))))
        # 更新优先经验回放PER中的优先级
        # self.replay_buffer.batch_update(tree_idx, abs_error.detach().numpy())
        ##################

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        ##################
        # YOUR CODE HERE #
        logits = self.eval_net(observation).detach()  # (n_action, )
        if test:  # 测试时，不采用epsilon贪心
            action = torch.argmax(logits).item()
        else:  # 训练时，采用epsilon贪心
            if np.random.rand() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = torch.argmax(logits).item()

        return action
        ##################

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        # 更好的做法是用一个Trainer类，而不是把经验池等放在Agent类中
        # 先采集足够的经验
        print('Collecting transition...', end=' ')
        t0 = time.time()
        # while len(self.replay_buffer) < self.args.break_step:
        while (not self.replay_buffer.isfull()) and (self.replay_buffer.tree.data_pointer < self.args.break_step):
            state = self.env.reset()
            self.init_game_setting()
            done = False
            while not done:
                # self.env.render()
                action = self.make_action(state, test=False)
                # print(action)
                state_new, reward, done, info = self.env.step(action)
                # 收集经验
                self.replay_buffer.push(state, action, reward, state_new, done)
                state = state_new
            # print(self.replay_buffer.tree.data_pointer, self.args.break_step)
        print('Time: {:.2f}m'.format((time.time()-t0)/60))

        print('Training...')
        rewards = []
        avg_rewards = []
        t0 = time.time()
        t_start = t0
        for i in range(self.args.num_episodes):
            state = self.env.reset()
            self.init_game_setting()
            self.epsilon = self.epsilons[i]
            done = False
            episode_reward = 0.0
            while not done:
                # self.env.render()
                action = self.make_action(state, test=False)
                state_new, reward, done, info = self.env.step(action)
                episode_reward += reward
                # 收集经验
                self.replay_buffer.push(state, action, reward, state_new, done)
                state = state_new
                self.train()
            rewards.append(episode_reward)
            avg_rewards.append(np.mean(rewards[-100:]))
            # for j in range(self.args.repeats):
            #     self.train()
            print('episode: {}, return: {:.2f}  epsilon: {:.3f}  time:{:.2f}m'
                  .format(i, episode_reward, self.epsilon, (time.time() - t0) / 60))
            t0 = time.time()

            if (i+1) % 100:
                # 保存训练数据
                data = {'res': rewards, 'avg_res': avg_rewards}
                file_name = self.args.save_dir + self.args.exp_name + '.pkl'
                with open(file_name, 'wb') as fp:
                    pickle.dump(data, fp)

                # 保存训练模型
                torch.save(self.eval_net.state_dict(), self.args.model_dir + self.args.exp_name + '.pt')

        print('Time: {:.2f}m'.format((time.time() - t_start) / 60))

        # # 保存训练数据
        # file_name = self.args.save_dir + self.args.exp_name + '.pkl'
        # with open(file_name, 'wb') as fp:
        #     pickle.dump(np.array(rewards), fp)

        # 保存训练模型
        torch.save(self.eval_net.state_dict(), self.args.model_dir + self.args.exp_name + '.pt')
        ##################
