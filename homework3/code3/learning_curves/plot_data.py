import pickle
import matplotlib.pyplot as plt
import numpy as np
import os


def PLOT_MATD3():
    save_dir = './png/'
    file_name = '../result/MATD3/MATD3.pkl'
    # file_name = '../result/MATD3/MATD3_no_reg.pkl'
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)

    res = data['res']
    avg_res = data['avg_res']
    step = np.arange(1, len(res) + 1)

    # axes = plt.axes()
    # axes.set_ylim([np.min(res) - 200, np.max(res) + 50])
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.plot(step, res)
    plt.plot(step, avg_res)
    legend_2 = 'Average of the last 100 episodes '
    plt.legend(['Reward', legend_2], loc=4)
    plt.grid()
    plt.title('with regularization')
    # plt.savefig(save_dir+'fig.png')
    plt.show()


def PLOT_VDN():
    save_dir = './png/'
    file_name = '../result/VDN/VDN.pkl'
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)

    res = data['res']
    avg_res = data['avg_res']
    step = np.arange(1, len(res) + 1)

    # axes = plt.axes()
    # axes.set_ylim([np.min(res) - 200, np.max(res) + 50])
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.plot(step, res)
    plt.plot(step, avg_res)
    legend_2 = 'Average of the last 100 episodes '
    plt.legend(['Reward', legend_2], loc=4)
    plt.grid()
    # plt.savefig(save_dir+'fig.png')
    plt.show()


def PLOT_QMIX():
    save_dir = './png/'
    # file_name = '../result/QMIX/QMIX.pkl'
    # file_name = '../result/QMIX2/QMIX2.pkl'
    file_name = '../result/QMIX3/QMIX3.pkl'
    with open(file_name, 'rb') as fp:
        data = pickle.load(fp)

    res = data['res']
    avg_res = data['avg_res']
    step = np.arange(1, len(res) + 1)

    # axes = plt.axes()
    # axes.set_ylim([np.min(res) - 200, np.max(res) + 50])
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.plot(step, res)
    plt.plot(step, avg_res)
    legend_2 = 'Average of the last 100 episodes '
    plt.legend(['Reward', legend_2], loc=4)
    plt.grid()
    plt.title('obs + action + rnn')
    # plt.savefig(save_dir+'fig.png')
    plt.show()


if __name__ == '__main__':
    # PLOT_MATD3()
    # PLOT_VDN()
    PLOT_QMIX()