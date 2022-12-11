#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pricing_project
@File    ：CTRL_Agent.py
@Author  ：xxuanzhu
@Date    ：2022/11/30 19:39 
@Purpose :
'''
import os
import sys
import time
from copy import deepcopy
import torch.nn as nn
import torch
from torch import Tensor

from agents.AgentBase import AgentBase
from config import *
from env_cityflow import CityFlowEnvM
import logging
import numpy as np

parent_dir = os.path.abspath(os.path.dirname(os.getcwd()))

today_time = time.strftime("%Y_%m_%d", time.localtime())
logging.basicConfig(level=logging.WARNING,
                    filename=parent_dir + '/log/train_CTRL_' + today_time + '.log',
                    filemode='a',
                    format='%(message)s')


class ReplayBuffer:
    def __init__(self, max_size: int, state_dim: int, action_dim: int, gpu_id: int = 0):
        self.p = 0  # pointer
        self.if_full = False
        self.cur_size = 0
        self.max_size = max_size

        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.states = torch.empty((max_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((max_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)
        self.undones = torch.empty((max_size, 1), dtype=torch.float32, device=self.device)

    def update(self, items: [Tensor]):
        states, actions, rewards, undones = items
        p = self.p + rewards.shape[0]  # pointer
        if p > self.max_size:
            self.if_full = True

            p0 = self.p
            p1 = self.max_size
            p2 = self.max_size - self.p
            p = p - self.max_size

            self.states[p0:p1], self.states[0:p] = states[:p2], states[-p:]
            self.actions[p0:p1], self.actions[0:p] = actions[:p2], actions[-p:]
            self.rewards[p0:p1], self.rewards[0:p] = rewards[:p2], rewards[-p:]
            self.undones[p0:p1], self.undones[0:p] = undones[:p2], undones[-p:]
        else:
            self.states[self.p:p] = states
            self.actions[self.p:p] = actions
            self.rewards[self.p:p] = rewards
            self.undones[self.p:p] = undones
        self.p = p
        self.cur_size = self.max_size if self.if_full else self.p

    def sample(self, batch_size: int) -> [Tensor]:
        ids = torch.randint(self.cur_size - 1, size=(batch_size,), requires_grad=False)
        return self.states[ids], self.actions[ids], self.rewards[ids], self.undones[ids], self.states[ids + 1]


def get_rewards_and_steps(env, actor, if_render: bool = False) -> (float, int):
    device = next(actor.parameters()).device

    state = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0
    for episode_steps in range(10000):
        tensor_state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        cumulative_returns += reward

        if if_render:
            env.render()
        if done:
            break
    return cumulative_returns, episode_steps + 1


class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd  # work directory
        self.env_eval = eval_env  # evaluate environment
        self.eval_step = 0  # evaluate step
        self.total_step = 0  # total step，每次累加 horizon_len个step
        self.start_time = time.time()
        self.eval_times = eval_times  # number of times that get episodic cumulative return
        self.eval_per_step = eval_per_step  # evaluate the agents per training steps

        self.recorder = []
        # print(f"\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
        #       f"\n| `time`: Time spent from the start of training to this moment."
        #       f"\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
        #       f"\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
        #       f"\n| `avgS`: Average of steps in an episode."
        #       f"\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
        #       f"\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
        #       f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'objC':>8}  {'objA':>8}")
        logging.warning(f"\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
                        f"\n| `time`: Time spent from the start of training to this moment."
                        f"\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
                        f"\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
                        f"\n| `avgS`: Average of steps in an episode."
                        f"\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
                        f"\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
                        f"\n| {'step':>8}  {'time':>8}  | {'avgR':>8}  {'stdR':>6}  {'avgS':>6}  | {'objC':>8}  {'objA':>8}")

    def evaluate_and_save(self, actor, horizon_len: int, logging_tuple: tuple):
        self.total_step += horizon_len
        if self.eval_step + self.eval_per_step > self.total_step:
            return
        self.eval_step = self.total_step

        rewards_steps_ary = [get_rewards_and_steps(self.env_eval, actor) for _ in range(self.eval_times)]
        rewards_steps_ary = np.array(rewards_steps_ary, dtype=np.float32)
        avg_r = rewards_steps_ary[:, 0].mean()
        std_r = rewards_steps_ary[:, 0].std()
        avg_s = rewards_steps_ary[:, 1].mean()

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        logging.warning(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
                        f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
                        f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")


def train_agent(args):
    args.init_before_training()
    gpu_id = 0

    env = args.env_class
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.states = env.reset()[np.newaxis, :]
    buffer = ReplayBuffer(gpu_id=gpu_id, max_size=args.buffer_size, state_dim=args.state_dim,
                          action_dim=args.action_dim)
    buffer_items = agent.explore_env(env, args.horizon_len * args.eval_times, if_random=True)
    buffer.update(buffer_items)

    evaluator = Evaluator(eval_env=env,
                          eval_per_step=args.eval_per_step,
                          eval_times=args.eval_times,
                          cwd=args.cwd
                          )

    torch.set_grad_enabled(False)
    while True:
        buffer_items = agent.explore_env(env, args.horizon_len)
        buffer.update(buffer_items)

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            torch.save(agent.act.state_dict(), 'pkls/ctrl_actor_parameters_' + today_time + '.pth')
            torch.save(agent.cri.state_dict(), 'pkls/ctrl_critic_parameters_' + today_time + '.pth')
            logging.warning("The training is finished")
            break  # stop training when reach `break_step` or `mkdir cwd/stop`


def build_mlp(dims: [int]) -> nn.Sequential:  # MLP (MultiLayer Perceptron)
    net_list = list()
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)


class ActorSAC(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.enc_s = build_mlp(dims=[state_dim, *dims])  # encoder of state
        self.dec_a_avg = build_mlp(dims=[dims[-1], action_dim])  # decoder of action mean
        self.dec_a_std = build_mlp(dims=[dims[-1], action_dim])  # decoder of action log_std
        self.log_sqrt_2pi = np.log(np.sqrt(2 * np.pi))
        self.soft_plus = nn.Softplus()

    def forward(self, state: Tensor) -> Tensor:
        state_tmp = self.enc_s(state)  # temporary tensor of state
        return self.dec_a_avg(state_tmp).tanh()  # action

    def get_action(self, state: Tensor) -> Tensor:  # for exploration
        state_tmp = self.enc_s(state)  # temporary tensor of state
        action_avg = self.dec_a_avg(state_tmp)
        action_std = self.dec_a_std(state_tmp).clamp(-20, 2).exp()

        noise = torch.randn_like(action_avg, requires_grad=True)
        action = action_avg + action_std * noise
        return action.clip(-1.0, 1.0)  # action (re-parameterize)

    def get_action_logprob(self, state: Tensor) -> [Tensor, Tensor]:
        state_tmp = self.enc_s(state)  # temporary tensor of state
        action_log_std = self.dec_a_std(state_tmp).clamp(-20, 2)
        action_std = self.dec_a_std(state_tmp).clamp(-20, 2).exp()
        action_avg = self.dec_a_avg(state_tmp)

        '''add noise to a_noise in stochastic policy'''
        noise = torch.randn_like(action_avg, requires_grad=True)
        a_noise = action_avg + action_std * noise

        '''compute log_prob according to mean and std of a_noise (stochastic policy)'''
        # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        log_prob = action_log_std + self.log_sqrt_2pi + noise.pow(2) * 0.5

        '''fix log_prob by adding the derivative of y=tanh(x)'''
        log_prob += (np.log(2.) - a_noise - self.soft_plus(-2. * a_noise)) * 2.  # better than below
        return a_noise.tanh(), log_prob.sum(1, keepdim=True)


class CriticTwin(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.enc_sa = build_mlp(dims=[state_dim + action_dim, *dims])  # encoder of state and action
        self.dec_q1 = build_mlp(dims=[dims[-1], action_dim])  # decoder of Q value 1
        self.dec_q2 = build_mlp(dims=[dims[-1], action_dim])  # decoder of Q value 2

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        sa_tmp = self.enc_sa(torch.cat((state, action), dim=1))
        return self.dec_q1(sa_tmp)  # Q value

    def get_q1_q2(self, state, action):
        sa_tmp = self.enc_sa(torch.cat((state, action), dim=1))
        return self.dec_q1(sa_tmp), self.dec_q2(sa_tmp)  # two Q values



class AgentSAC(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: CTRLConfig = CTRLConfig()):
        self.act_class = getattr(self, 'act_class', ActorSAC)  # get the attribute of object `self`
        self.cri_class = getattr(self, 'cri_class', CriticTwin)  # get the attribute of object `self`
        super().__init__(net_dims, state_dim, action_dim, gpu_id, args)
        self.cri_target = deepcopy(self.cri)  # 孪生Q网络

        self.alpha_log = torch.tensor(-1, dtype=torch.float32, requires_grad=True, device=self.device)  # trainable var
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=args.learning_rate)
        self.target_entropy = np.log(action_dim)

    def explore_env(self, env, horizon_len: int, if_random: bool = False) -> [Tensor]:
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.states[0]
        get_action = self.act.create_action
        for i in range(horizon_len):
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device)
            action = torch.rand(self.action_dim) * 2 - 1.0 if if_random else get_action(state.unsqueeze(0))[0]

            states[i] = state
            actions[i] = action

            ary_action = action.detach().cpu().numpy()
            ary_state, reward, done, _ = env.step(ary_action)
            if done:
                ary_state = env.reset()

            rewards[i] = reward
            dones[i] = done

        self.states[0] = ary_state
        rewards = rewards.unsqueeze(1)
        undones = (1.0 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, rewards, undones

    def update_net(self, buffer) -> [float]:
        obj_critics = obj_actors = 0.0
        update_times = int(buffer.cur_size * self.repeat_times / self.batch_size)
        for i in range(update_times):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)
            obj_critics += obj_critic.item()

            action, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (self.alpha_log * (logprob - self.target_entropy).detach()).mean()
            self.optimizer_update(self.alpha_optim, obj_alpha)

            alpha = self.alpha_log.exp()
            obj_actor = (self.cri(state, action) + logprob * alpha).mean()
            self.optimizer_update(self.act_optimizer, -obj_actor)
            obj_actors += obj_actor.item()
        return obj_critics / update_times, obj_actors / update_times

    def get_obj_critic(self, buffer, batch_size: int) -> (Tensor, Tensor):
        with torch.no_grad():
            state, action, reward, undone, next_state = buffer.sample(batch_size)

            next_action, next_log_prob = self.act.get_action_logprob(next_state)  # stochastic policy
            next_q = torch.min(*self.cri_target.get_q1_q2(next_state, next_action))  # twin critics
            alpha = self.alpha_log.exp()
            q_label = reward + undone * self.gamma * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = (self.criterion(q1, q_label) + self.criterion(q2, q_label)) / 2.
        return obj_critic, state

def train_sac_for_CTRL(gpu_id=0):
    env_class = CityFlowEnvM()
    env_args = {
        'env_name': 'cityflow',  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': env_class.edges_num,
        # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': env_class.edges_num,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }

    args = CTRLConfig(agent_class=AgentSAC, env_class=CityFlowEnvM, env_args=env_args)
    args.break_step = int(8e5)
    args.net_dims = (64, 32)
    args.gpu_id = gpu_id
    args.gamma = 0.97

    train_agent(args)


if __name__ == '__main__':
    train_sac_for_CTRL(gpu_id=int(sys.argv[1]) if len(sys.argv) > 1 else -1)
