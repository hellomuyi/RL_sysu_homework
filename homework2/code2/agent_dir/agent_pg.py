import os
import random
import copy
import time
import pickle
import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn, optim
from agent_dir.agent import Agent


class PGNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PGNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        # 两层全连接网络
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        ##################

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        inputs = torch.from_numpy(inputs).to(torch.float32)
        return self.net(inputs)
        ##################

class BaselineNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BaselineNetwork, self).__init__()
        ##################
        # YOUR CODE HERE #
        # 两层全连接网络
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # 基线网络
        self.v_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        ##################

    def forward(self, inputs):
        ##################
        # YOUR CODE HERE #
        inputs = torch.from_numpy(inputs).to(torch.float32)
        return self.net(inputs)
        ##################

class AgentPG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentPG, self).__init__(env)
        ##################
        # YOUR CODE HERE #
        self.args = args
        self.env = env

        o_dim = env.observation_space.shape[0]  # （210,160,3） --> (4, 84, 84)
        self.action_dim = env.action_space.n  # 6
        self.net = PGNetwork(o_dim, args.hidden_size, self.action_dim)
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr)
        if self.args.test:  # 测试时加载已经训练好的模型
            model_path = self.args.model_dir + self.args.exp_name + '.pt'
            self.net.load_state_dict(torch.load(model_path))
        if self.args.with_baseline:
            self.v_net = BaselineNetwork(o_dim, args.hidden_size)
            self.optim_v = optim.Adam(self.v_net.parameters(), lr=args.lr)
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

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        obs_batch = np.array(self.eps_obs)
        act_batch = torch.LongTensor(np.array(self.eps_act))
        discounted_eps_res = torch.FloatTensor(self.discounted_eps_res)

        logits = self.net(obs_batch)
        log_p = torch.log_softmax(logits, dim=-1)
        log_p = log_p[torch.arange(len(act_batch)), act_batch]
        # loss = torch.mean(-log_p * discounted_eps_res)
        if self.args.with_baseline:
            v = self.v_net(obs_batch).squeeze()
            delta = discounted_eps_res-v
            # 更新基线网络
            loss_v = torch.mean(delta ** 2)
            self.optim_v.zero_grad()
            loss_v.backward()
            nn.utils.clip_grad_norm_(self.v_net.parameters(), self.args.grad_norm_clip)
            self.optim_v.step()
        else:
            delta = discounted_eps_res
        # 更新策略网络
        loss = torch.mean(-log_p * delta.detach())
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.args.grad_norm_clip)
        self.optim.step()
        ##################

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return: action
        """
        ##################
        # YOUR CODE HERE #
        logits = self.net(observation).detach()
        p = torch.softmax(logits, -1)
        # p = torch.squeeze(p, dim=1)
        if test:
            action = torch.argmax(p).item()
        else:
            action = torch.multinomial(p, 1).item()
        return action
        # return self.env.action_space.sample()
        ##################

    def run(self):
        """
        Implement the interaction between agent and environment here
        """
        ##################
        # YOUR CODE HERE #
        print('Training...')
        rewards = []
        t0 = time.time()
        t_start = t0
        for i in range(self.args.num_episodes):
            self.eps_obs = []  # 一个episode的所有状态
            self.eps_res = []  # 一个episode的所有奖励
            self.eps_act = []  # 一个episode的所有动作

            state = self.env.reset()
            self.init_game_setting()
            done = False
            episode_reward = 0.0

            while not done:
                action = self.make_action(state, test=False)
                state_new, reward, done, info = self.env.step(action)
                episode_reward += reward
                # 收集经验
                self.eps_obs.append(state)
                self.eps_act.append(action)
                self.eps_res.append(reward)
                state = state_new
            rewards.append(episode_reward)

            # 计算一个episode内各时刻的折扣回报
            discounted_eps_res = np.zeros_like(self.eps_res)
            delta = 0
            # 从后往前，递推地计算各个时刻的累积回报Gt
            for t in reversed(range(len(self.eps_res))):
                delta = delta * self.args.gamma + self.eps_res[t]
                discounted_eps_res[t] = delta

            self.discounted_eps_res = discounted_eps_res
            if not self.args.with_baseline:
                # 标准化G
                # 我们希望G值有正有负，这样比较容易学习。
                discounted_eps_res -= np.mean(discounted_eps_res)
                discounted_eps_res /= np.std(discounted_eps_res)
                self.discounted_eps_res = discounted_eps_res

            self.train()
            print('episode: {}, return: {:.2f}  time:{:.2f}m'
                  .format(i, episode_reward, (time.time() - t0) / 60))
            t0 = time.time()
        print('Time: {:.2f}m'.format((time.time() - t_start) / 60))

        # 保存训练数据
        file_name = self.args.save_dir + self.args.exp_name + '.pkl'
        with open(file_name, 'wb') as fp:
            pickle.dump(np.array(rewards), fp)

        # 保存训练模型
        torch.save(self.net.state_dict(), self.args.model_dir + self.args.exp_name + '.pt')
        ##################
