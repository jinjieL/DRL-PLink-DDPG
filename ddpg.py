# -*- coding: utf-8 -*-

import os, random
import numpy as np

# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
# import topo3_env
from utils.OU import OUNoise

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''


rl_mu       = 0.001    # actor_learning_rate
rl_q        = 0.001     # critic_learning_rate
tau         = 0.05   # target smoothing coefficient
target_update_interval = 10
gamma       = 0.99   # discounted factor
capacity    = 5000   # replay buffer size
batch_size  = 16      # mini batch size
update_iteration    = 10

seed        = False
random_seed = 9527


device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)

if seed:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

max_action = 1
min_Val = torch.tensor(1e-7).float().to(device) # min value

algorithm    = 'DRL_PLink'
model = 'model'
data = 'data'
directory = './' + model + '/' + algorithm + '/'
directory_data = './' + data + '/' + algorithm + '/'


def softmax(X):
    return np.exp(X) / np.sum(np.exp(X))


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r = [], [], [], []

        for i in ind:
            X, Y, U, R = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, rate_dim, path_dim, action_dim, kPath, init_w=3e-3):
        super(Actor, self).__init__()
        self.kPath = kPath
        self.rate_dim = rate_dim
        self.path_dim = path_dim
        self.action_dim = action_dim

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 64)

        self.l4_r = nn.Linear(64, 10)
        # self.r0 = NoisyLinear(10, 1)
        # self.r1 = NoisyLinear(10, 1)
        # self.r2 = NoisyLinear(10, 1)
        self.r = []
        for i in xrange(rate_dim):
            self.r.append(nn.Linear(10, 1))

        self.l4MF = nn.Linear(64, 10)
        self.l4EF1 = nn.Linear(64, 10)
        self.l4EF2 = nn.Linear(64, 10)

        self.MF = []
        self.EF1 = []
        self.EF2 = []
        for i in xrange(path_dim/3):
            self.MF.append(nn.Linear(10, kPath))
            self.EF1.append(nn.Linear(10, kPath))
            self.EF2.append(nn.Linear(10, kPath))
        # self.lMF.weight.data.uniform_(-init_w, init_w)
        # self.lMF.bias.data.uniform_(-init_w, init_w)
        # self.lEF1.weight.data.uniform_(-init_w, init_w)
        # self.lEF1.bias.data.uniform_(-init_w, init_w)
        # self.lEF2.weight.data.uniform_(-init_w, init_w)
        # self.lEF2.bias.data.uniform_(-init_w, init_w)
        # self.l4_r.weight.data.uniform_(-init_w, init_w)
        # self.l4_r.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        shape = x.shape
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        l4_r = F.relu(self.l4_r(x))
        # r0 = F.sigmoid(self.r0(l4_r))
        # r1 = F.sigmoid(self.r1(l4_r))
        # r2 = F.sigmoid(self.r2(l4_r))
        # r = torch.cat([r0, r1, r2], 1)
        # r = F.softmax(r, dim=1)
        # r = F.softmax(self.r0(l4_r), dim=1)
        Rout = []
        for i in xrange(self.rate_dim):
            rate = F.sigmoid(self.r[i](l4_r))
            Rout.append(rate)
        print Rout

        oRate = torch.cat([Rout[3*0+0],Rout[3*0+1],Rout[3*0+2]], 1)
        oRate = F.softmax(oRate, dim=1)

        for i in xrange(1, self.rate_dim/3):
            r = torch.cat([Rout[3*i+0],Rout[3*i+1],Rout[3*i+2]], 1)
            r = F.softmax(r, dim=1)
            oRate = torch.cat([oRate, r], 1)

        print oRate

        MF = F.relu(self.l4MF(x))
        EF1 = F.relu(self.l4EF1(x))
        EF2 = F.relu(self.l4EF2(x))

        MFout = []
        EF1out = []
        EF2out = []

        a = F.softmax(self.MF[0](MF), dim=1)
        print a.shape
        print torch.tensor(a.max(1)[1]).float()

        for i in xrange(self.path_dim/3):
            MFout.append(F.softmax(self.MF[i](MF), dim=1).max(1)[1].float())
            EF1out.append(F.softmax(self.EF1[i](EF1), dim=1).max(1)[1].float())
            EF2out.append(F.softmax(self.EF2[i](EF2), dim=1).max(1)[1].float())

        # pMF = self.kPath * F.sigmoid(self.lMF(x))
        # pEF1 = self.kPath * F.sigmoid(self.lEF1(x))
        # pEF2 = self.kPath * F.sigmoid(self.lEF2(x))

        #oMF = MFout[0]
        #oEF1 = EF1out[0]
        #oEF2 = EF2out[0]

        oMF = MFout[0].view(shape[0],-1)
        oEF1 = EF1out[0].view(shape[0],-1)
        oEF2 = EF2out[0].view(shape[0],-1)

        for i in xrange(1, self.path_dim/3):
            oMF = torch.cat([oMF, MFout[i].view(shape[0],-1)], 1)
            oEF1 = torch.cat([oEF1, EF1out[i].view(shape[0],-1)], 1)
            oEF2 = torch.cat([oEF2, EF2out[i].view(shape[0],-1)], 1)

        out = torch.cat([oRate, oMF, oEF1, oEF2], 1)
        print out
        return out


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, action_dim)
        self.l3 = nn.Linear(2 * action_dim, 8)
        self.l4 = nn.Linear(8, 1)

        self.l4.weight.data.uniform_(-init_w, init_w)
        self.l4.bias.data.uniform_(-init_w, init_w)

    def forward(self, s, a):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        # print x
        # print a
        x = F.relu(self.l3(torch.cat([x, a], 1)))
        x = self.l4(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, rate_dim, path_dim, action_dim, kPath):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.kPath = kPath

        self.actor = Actor(state_dim, rate_dim, path_dim, action_dim, kPath).to(device)
        self.actor_target = Actor(state_dim, rate_dim, path_dim, action_dim, kPath).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), rl_mu)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), rl_q)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_origin = self.actor(state).cpu().data.numpy().flatten()
        # print action_origin
        return action_origin

    def select_action_with_epsilon(self, state, epsilon):
        if random.random() < epsilon:
            print '探索explore'
            r1 = np.random.rand()
            r2 = np.random.rand()
            r3 = np.random.rand()
            r = softmax(np.array([r1, r2, r3]))
            path = np.random.rand(self.action_dim-3) * self.kPath
            action = np.hstack((r, path))
        else:
            print '利用exploit'
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action_origin = self.actor(state).cpu().data.numpy().flatten()
            action = action_origin
        return action

    def select_noise_action(self, state, nodes):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action_origin = self.actor(state).cpu().data.numpy().flatten()
        print 'action_origin', action_origin
        ou = OUNoise(len(action_origin))
        noise = ou.noise()
        # print noise

        action_noise = action_origin + noise
        # print action_noise
        action_noise[0:3] = softmax(action_noise[0:3])
        # print action_noise

        rate = action_noise[0:3]

        adim = (nodes**2-nodes)/2
        pMF = action_noise[3:adim + 3]
        pEP1 = action_noise[adim + 3:adim*2 + 3]
        pEP2 = action_noise[adim*2 + 3:adim*3 + 3]
        # action = np.vstack((rate, path))
        action = np.hstack((rate, pMF, pEP1, pEP2))
        return action

    def update(self):

        for it in range(update_iteration):
            # Sample replay buffer
            x, y, u, r = self.replay_buffer.sample(batch_size)
            # print x
            # print y
            # print u   # [[0.26312112 0.46716092 0.26971796 1.30174015 0.70692096 1.19027859] [0.26158407 0.37091041 0.36750552 1.0430283  1.11125301 1.08606328]]
            # print r   # [[-0.7705758] [-0.7705758]]

            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.FloatTensor(u).to(device)
            reward = torch.FloatTensor(r).to(device)
            # print reward  # tensor([[ 0.1377],[-0.0596]], device='cuda:0')
            print next_state


            # Compute the target Q value
            target_action = self.actor_target(next_state)
            # print target_action
            # tensor([[0.3146, 0.3404, 0.3450, 0.9705, 0.8856, 1.1336],
            #         [0.3147, 0.3404, 0.3450, 0.9705, 0.8859, 1.1338]], device='cuda:0',
            #        grad_fn= < CatBackward >)
            target_Q = self.critic_target(next_state, target_action)
            target_Q = reward + (gamma * target_Q).detach()
            # print target_Q
            # Get current Q estimate
            current_Q = self.critic(state, action)
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            # Compute actor loss
            current_action = self.actor(state)
            # print current_action
            actor_loss = -self.critic(state, current_action).mean()
            self.writer.add_scalar('actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
            # print 'update'

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.plk')
        torch.save(self.critic.state_dict(), directory + 'critic.plk')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.plk'))
        self.critic.load_state_dict(torch.load(directory + 'critic.plk'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


