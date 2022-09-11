#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import numpy as np
from pylab import *
import math

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

device = T.device("cuda:0")


# device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
# print(device)
# -----------------    REINFORCE with nonlinear approximators    -----------------#

class approximator(nn.Module):
    def __init__(self, paras):
        super(approximator, self).__init__()
        # 定义层
        self.num_inputs = paras['num_state']  # 输入有40维
        self.num_outputs = paras['num_action']  # 输出有input_dim 10维
        self.alpha_policy = paras['alpha_policy']

        self.l1 = nn.Linear(self.num_inputs, 1200)  # （10 * 4， 36）
        self.l2 = nn.Linear(1200, 360)  # （36， 18）
        self.l3 = nn.Linear(360, self.num_outputs)  # 输出，(18， 10)

        self.l3.weight.data.mul_(0.1)
        self.l3.bias.data.mul_(0.0)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha_policy)



    def forward(self, x):
        x = T.from_numpy(x)
        x = x.float()
        # x = x.view(1, 40)
        x = Variable(x.cuda(), requires_grad=True)
        x = T.tanh(self.l1(x))
        x = T.tanh(self.l2(x))
        mu = T.tanh(self.l3(x))
        # mu = self.l3(x)
        # sigma = T.exp(self.l4(x))
        # return mu, sigma
        logstd = T.zeros_like(mu)
        std = T.exp(logstd)
        return mu, std, logstd


# 思维决策
class REINFORCE_NN(object):
    # 初始化
    def __init__(self,
                 paras={}):
        super(REINFORCE_NN, self).__init__()
        self.state_dim = paras['num_state']
        self.action_dim = paras['num_action']
        self.lower_bound_action = paras['lower_bound_action']
        self.upper_bound_action = paras['upper_bound_action']

        self.gamma = 1
        self.approximator = approximator(paras)
        self.approximator = self.approximator.to(device)

        self.loss = 0

        # 选择行为

    def get_actions(self, state):
        mu, std, logstd = self.approximator.forward(state)
        action = T.normal(mu, std)
        action = action.cpu().data.numpy()
        action = np.clip(action, self.lower_bound_action, self.upper_bound_action)  # 小于lower的变成lower，大于upper的变成upper
        return action

    def log_density(self, x, mu, std, logstd):
        var = std.pow(2)
        log_density = -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
        return log_density.sum(1, keepdim=True)

    def get_returns(self, rewards):
        G = T.from_numpy(
            np.array([sum([rewards[k] * (self.gamma ** (k - t - 1)) for k in range(t + 1, len(rewards))]) for t in
                      range(len(rewards))]))

        gammas = T.from_numpy(np.array([self.gamma ** t for t in range(len(rewards))]))

        G = G * gammas.double()
        G = (G - G.mean()) / G.std()
        return G.int()

    def get_loss(self, returns, states, actions):
        s = np.array([list(states[idx][0]) for idx in range(len(states))])
        a = np.array([actions[idx][0] for idx in range(len(actions))])
        mu, std, logstd = self.approximator.forward(s)
        mu, std, logstd = mu.data.cpu(), std.data.cpu(), logstd.data.cpu()
        log_policy = self.log_density(T.Tensor(a), mu, std, logstd)
        returns = returns.unsqueeze(1)

        objective = returns.float() * log_policy
        objective = objective.mean()
        return -objective

    def train(self, returns, states, actions):
        self.loss = Variable(self.get_loss(returns, states, actions).cuda(), requires_grad=True)
        self.approximator.optimizer.zero_grad()
        self.loss.sum().backward()
        self.approximator.optimizer.step()

    # 学习更新参数
    def train_model(self, memory):
        memory = np.array(memory)
        # states = np.vstack(memory[:, 0])
        states = list(memory[:, 0])
        actions = list(memory[:, 1])
        rewards = list(memory[:, 2])
        # masks = list(memory[:, 3])

        returns = self.get_returns(rewards)
        self.train(returns, states, actions)
        return returns
