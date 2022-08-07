
import numpy as np
import pandas as pd


class Sarsa:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.2):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        ''' build q table'''
        ############################

        # YOUR IMPLEMENTATION HERE #
        np.random.seed(1)
        obs_dim = 4 * 4  # 观测空间为16
        self.q = np.random.rand(obs_dim, len(actions))
        ############################

    def choose_action(self, observation):
        ''' choose action from q table '''
        ############################

        # YOUR IMPLEMENTATION HERE #
        # epsilon 贪心
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(self.actions))
        else:
            # 由观测坐标换算为状态序号，agent左上角坐标为(x=observation[0], y=observation[1])
            # 换算公式为index = (x-5)//40 *4 + (y-5)//40
            x, y = int(observation[0]), int(observation[1])  # 提取observation的左上角坐标
            s_idx = (x - 5) // 40 + (y - 5) // 40 * 4  # 换算为Q表的行下标
            action = np.argmax(self.q[s_idx, :])
        return action
        ############################

    def learn(self, s, a, r, s_):
        ''' update q table '''
        ############################

        # YOUR IMPLEMENTATION HERE #
        x, y = int(s[0]), int(s[1])  # 提取s的左上角坐标
        idx_s = (x - 5) // 40 + (y - 5) // 40 * 4  # 换算为Q表的行下标
        if self.check_state_exist(s_):
            td_target = r + self.gamma * 0  # 终止状态的Q值为0
        else:
            x_, y_ = int(s_[0]), int(s_[1])  # 提取s_的左上角坐标
            idx_s_ = (x_ - 5) // 40 + (y_ - 5) // 40 * 4  # 换算为Q表的行下标
            if np.random.rand() < self.epsilon:
                action = np.random.choice(len(self.actions))
                td_target = r + self.gamma * (self.q[idx_s_, action])
            else:
                td_target = r + self.gamma * np.max(self.q[idx_s_, :])
        self.q[idx_s, a] = self.q[idx_s, a] + self.lr * (td_target - self.q[idx_s, a])
        ############################

    def check_state_exist(self, state):
        ''' check state '''
        ############################

        # YOUR IMPLEMENTATION HERE #
        if state == 'terminal':
            return True
        else:
            return False
        ############################