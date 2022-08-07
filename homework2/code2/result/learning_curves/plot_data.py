import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def smooth(data, wd=2):
    """
    :param data: ndarray，一维或二维
    :param wd:
    :return:
    """
    if not (isinstance(wd, int) and wd > 0):
        raise ValueError('wd must be a positive integer')
    elif 1 == wd:
        return data
    else:
        weight = np.ones(wd) / wd
        if 1 == data.ndim:
            return np.convolve(weight, data, "valid")
        elif 2 == data.ndim:
            smooth_data = []
            for d in data:
                d = np.convolve(weight, d, "valid")
                smooth_data.append(d)
            return np.array(smooth_data)
        else:
            raise ValueError('data must be a one-dimensional or two-dimensional ndarray')


def PLOT_DQN():
    file_name = './nature_dqn1.pkl'
    with open(file_name, 'rb') as fp:
        dqn_nature = pickle.load(fp)
    file_name = './double_dqn1.pkl'
    with open(file_name, 'rb') as fp:
        dqn_double = pickle.load(fp)
    file_name = './dueling_dqn1.pkl'
    with open(file_name, 'rb') as fp:
        dqn_dueling = pickle.load(fp)

    res_nature = dqn_nature['res']
    res_double = dqn_double['res']
    res_dueling = dqn_dueling['res']

    wd = 10
    res_nature = smooth(np.array(res_nature), wd)
    res_double = smooth(np.array(res_double), wd)
    res_dueling = smooth(np.array(res_dueling), wd)

    step = np.arange(1, len(res_nature) + 1)

    axes = plt.axes()
    axes.set_ylim([np.min([res_nature, res_double, res_dueling]) - 5, np.max([res_nature, res_double, res_dueling]) + 5])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(step, res_nature)
    plt.plot(step, res_double)
    plt.plot(step, res_dueling)
    plt.legend(['nature_dqn', 'double_dqn', 'd3qn'], loc='upper left')
    plt.grid()
    plt.title('epsilon=0.01')
    # plt.savefig(save_dir+'fig.png')
    plt.show()


def PLOT_A2C():
    save_dir = './png/'
    file_name = '../A2C/A2C_CartPole-v0_seed1.pkl'
    # file_name = 'TD3_BipedalWalker.pkl'
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)

    res = data['res']
    avg_res = data['avg_res']
    step = np.arange(1, len(res) + 1)

    axes = plt.axes()
    # axes.set_ylim([np.min(res) - 200, np.max(res) + 50])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(step, res)
    plt.plot(step, avg_res)
    legend_2 = 'Running average of the last 100 episodes '
    plt.legend(['Reward', legend_2], loc=4)
    plt.grid()
    # plt.savefig(save_dir+'fig.png')
    plt.show()


def PLOT_TD3():
    save_dir = './png/'
    file_name = '../TD3_LunarLanderContinuous-v2/TD3.pkl'
    # file_name = 'TD3_BipedalWalker.pkl'
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)

    res = data['res']
    avg_res = data['avg_res']
    step = np.arange(1, len(res) + 1)

    axes = plt.axes()
    axes.set_ylim([np.min(res) - 200, np.max(res) + 50])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(step, res)
    plt.plot(step, avg_res)
    legend_2 = 'Running average of the last 100 episodes '
    plt.legend(['Reward', legend_2], loc=4)
    plt.grid()
    # plt.savefig(save_dir+'fig.png')
    plt.show()


def PLOT_TD3_2():
    save_dir = './png/'
    file_name = '../TD3_BipedalWalker/TD3_BipedalWalker.pkl'
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)

    res = data['res']
    avg_res = data['avg_res']
    step = np.arange(1, len(res) + 1)

    axes = plt.axes()
    # axes.set_ylim([np.min(res) - 10, np.max(res) + 50])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(step, res)
    plt.plot(step, avg_res)
    legend_2 = 'Running average of the last 100 episodes '
    plt.legend(['Reward', legend_2], loc=4)
    plt.grid()
    # plt.savefig(save_dir+'fig.png')
    plt.show()

if __name__ == '__main__':
    # PLOT_DQN()
    # PLOT_A2C()
    # PLOT_TD3()
    PLOT_TD3_2()