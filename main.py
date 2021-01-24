# -*- coding: utf-8 -*-
from env import NetworkEnv
from env_rest import NetworkEnvRestAPI
from ddpg import DDPG, Replay_buffer
# from OU import OUNoise

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pandas as pd
import math

max_episode = 100
max_step = 10
test_max_episode = 5
test_max_step = 10
algorithm    = 'DRL_PLink'
model = 'model'
data = 'data'
directory = './' + model + '/' + algorithm + '/'
directory_data = './' + data + '/' + algorithm + '/'
datamining = 'datamining.txt'
websearch = 'websearch.txt'

# epsilon_start = 1.0
# epsilon_final = 0.01
# epsilon_decay = 15
#
# epsilon_by_episode = lambda episode_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode_idx / epsilon_decay)
#
# plt.plot([epsilon_by_episode(i) for i in range(max_episode)])
# # plt.show()


def softmax(X):
    return np.exp(X) / np.sum(np.exp(X))


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        # self.low = action_space.low
        # self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    # def get_action(self, action, t=0):
    #     ou_state = self.evolve_state()
    #     self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
    #     return np.clip(action + ou_state, self.low, self.high)

# https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py


def plot(max_episode, rewards):
    plt.figure()
    # plt.subplot(131)
    plt.title('episode %s. reward: %s' % (max_episode, rewards[-1]))
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.plot(rewards, 'b*--')
    plt.savefig(directory + 'ddpg.png')
    # plt.show()


def Test():
    print 'test'
    cmd = 'sudo python ./utils/delete_episode_data.py'
    os.system(cmd)

    env = NetworkEnv()
    agent = DDPG(env.state_dim, env.rate_dim, env.path_dim, env.action_dim, env.kPath)
    rewards = []

    workload = datamining
    state = env.reset(workload)
    agent.load()
    t1 = time.time()
    for i in range(test_max_episode):

        Perform = []
        ep_r = 0.0
        if i%5==0:
            if env.CDF_file == datamining:
                env.set_CDF_file(websearch)
            else:
                env.set_CDF_file(datamining)

        for t in range(test_max_step):
            print '\n'
            print '第%d回合,第%d步骤:' % (i,t)
            print 'state', state

            action = agent.select_action(state)
            print 'action:', action

            # execute action
            next_state, reward, perform = env.step(action)
            Perform.append(perform)

            ep_r += reward
            state = next_state
            print 'reward', reward
            print("Ep_i {}, the ep_r is {:0.2f}, the step is {}, the reward is {}".format(i, ep_r, t, reward))
            if t == test_max_step - 1:
                print("Ep_i {}, the ep_r is {:0.2f}".format(i, ep_r))
                df_perform = pd.DataFrame(Perform)
                df_perform.to_csv("./data/test_perform.csv", mode='a', header=False, index=False)
                break

        rewards.append(ep_r)

    file_time = open('./data/test_time.txt', mode='w')
    df = pd.DataFrame([rewards])
    df.to_csv("./data/test_rewards.csv", mode='w', header=False, index=False)
    print max_episode, 'episode的总运行时间:', time.time() - t1
    file_time.write(str(time.time() - t1))


def main():
    cmd = 'sudo python ./utils/writeCSV.py'
    os.system(cmd)
    cmd = 'sudo python ./utils/write.py'
    os.system(cmd)

    env = NetworkEnv()
    agent = DDPG(env.state_dim, env.rate_dim, env.path_dim, env.action_dim, env.kPath)
    ou = OUNoise(env.action_dim)
    rewards = []
    workload = datamining

    state = env.reset(workload)
    # agent.load()
    t1 = time.time()

    for i in range(1, max_episode+1):
        # ou.reset()
        Perform = []
        ou.reset()
        ep_r = 0.0
        # epsilon = epsilon_by_episode(i)
        # print 'epsilon', epsilon

        if i%5==0:
            if env.CDF_file == datamining:
                env.set_CDF_file(websearch)
            else:
                env.set_CDF_file(datamining)

        for t in range(max_step):
            print '\n'
            print '第%d回合,第%d步骤:' % (i,t)
            print 'state', state

            # action = agent.select_action_with_epsilon(state, epsilon)
            action = agent.select_action(state)
            # action = agent.select_noise_action(state, env.nodes)
            print 'action:', action

            # issue 3 add noise to action
            noise = ou.evolve_state()
            # print noise
            action_noise = action + noise
            # print action_noise
            for j in xrange(env.rate_dim/3):
                # print action_noise[j*3:j*3+3]
                action_noise[j*3:j*3+3] = softmax(action_noise[j*3:j*3+3])
            # rate = action_noise[0:3]
            # adim = (env.nodes ** 2 - env.nodes) / 2
            # pMF = action_noise[3:adim + 3]
            # pEP1 = action_noise[adim + 3:adim * 2 + 3]
            # pEP2 = action_noise[adim * 2 + 3:adim * 3 + 3]
            # # action = np.vstack((rate, path))
            # action = np.hstack((rate, pMF, pEP1, pEP2))
            # print 'action_noise:', action_noise

            # execute action
            next_state, reward, perform = env.step(action)
            print 'reward', reward
            Perform.append(perform)

            # Add replay buffer
            agent.replay_buffer.push((state.flatten(), next_state.flatten(), action.flatten(), reward))

            ep_r += reward
            state = next_state

            print("Ep_i {}, the ep_r is {:0.2f}, the step is {}, the reward is {}".format(i, ep_r, t, reward))
            if len(agent.replay_buffer.storage) >= 64:
                print 'update'
                agent.update()
            if t == max_step - 1:
                print("Ep_i {}, the ep_r is {:0.2f}".format(i, ep_r))
                agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                df_perform = pd.DataFrame(Perform)
                df_perform.to_csv("./data/train_perform.csv", mode='a', header=False, index=False)
                break

        rewards.append(ep_r)
        if i % 2 == 0:
            plot(i, rewards)
            agent.save()

    file_time = open('./data/train_time.txt', mode='w')
    t_end = time.time()-t1
    print max_episode, 'episode的总运行时间:', t_end

    file_time.write(str(t_end) + '\n')
    agent.save()
    #Test()


if __name__ == '__main__':
    main()
    #cmd = 'sudo python write.py'
    #os.system(cmd)
    os.system('sudo mn -c')
    os.system('sudo ovs-vsctl --all destroy qos')
    os.system('sudo ovs-vsctl --all destroy queue')
    time.sleep(2)
    Test()
    time.sleep(5)
    os.system('sudo mn -c')
    os.system('sudo ovs-vsctl --all destroy qos')
    os.system('sudo ovs-vsctl --all destroy queue')

