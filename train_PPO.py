#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：new_goal 
@File    ：train_PPO.py
@Author  ：xxuanZhu
@Date    ：2022/10/13 20:33 
@Purpose :
'''

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath[0])
sys.path.append(rootPath)
import time
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.normal import Normal
import numpy as np

from AgentBase import AgentBase
from config import Config
from env_cityflow import CityFlowEnvM
import logging

parent_dir = os.path.abspath(os.path.dirname(os.getcwd()))



today_time = time.strftime("%Y_%m_%d", time.localtime())
logging.basicConfig(level=logging.WARNING,
                        filename=parent_dir+'/log/trainPPO_'+today_time+'.log',
                        filemode='a',
                        format='%(message)s')


class ActorPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, action_dim])
        self.action_std_log = nn.Parameter(torch.zeros((1, action_dim)), requires_grad=True)  # trainable parameter

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state).tanh()

    def get_action(self, state: Tensor) -> (Tensor, Tensor):  # for exploration
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        return action, logprob

    def get_logprob_entropy(self, state: Tensor, action: Tensor) -> (Tensor, Tensor):
        action_avg = self.net(state)
        action_std = self.action_std_log.exp()

        dist = Normal(action_avg, action_std)
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        return logprob, entropy

    @staticmethod
    def convert_action_for_env(action: Tensor) -> Tensor:
        return action.tanh()


class CriticPPO(nn.Module):
    def __init__(self, dims: [int], state_dim: int, _action_dim: int):
        super().__init__()
        self.net = build_mlp(dims=[state_dim, *dims, 1])

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)  # advantage value


def build_mlp(dims: [int]) -> nn.Sequential:
    net_list = []
    for i in range(len(dims) - 1):
        net_list.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
    del net_list[-1]  # remove the activation of output layer
    return nn.Sequential(*net_list)





def get_gym_env_args(env, if_print: bool) -> dict:
    env_name = env.env_name
    state_dim = env.state_dim
    action_dim = env.action_dim
    if_discrete = env.if_discrete

    env_args = {'env_name': env_name,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'if_discrete': if_discrete, }
    if if_print:
        env_args_str = repr(env_args).replace(',', f",\n{'':11}")
        print(f"env_args = {env_args_str}")
    return env_args


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect  # 获取对象信息
    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def build_env(env_class=None, env_args=None):
    env = env_class(**kwargs_filter(env_class.__init__, env_args.copy()))
    for attr_str in ('env_name', 'state_dim', 'action_dim', 'if_discrete'):
        setattr(env, attr_str, env_args[attr_str])
    return env





class AgentPPO(AgentBase):
    def __init__(self, net_dims: [int], state_dim: int, action_dim: int, gpu_id: int = 0, args: Config = Config()):
        self.if_off_policy = False
        self.act_class = getattr(self, "act_class", ActorPPO)
        self.cri_class = getattr(self, "cri_class", CriticPPO)
        AgentBase.__init__(self, net_dims, state_dim, action_dim, gpu_id, args)

        self.ratio_clip = getattr(args, "ratio_clip", 0.25)  # `ratio.clamp(1 - clip, 1 + clip)`
        self.lambda_gae_adv = getattr(args, "lambda_gae_adv", 0.95)  # could be 0.80~0.99
        self.lambda_entropy = getattr(args, "lambda_entropy", 0.01)  # could be 0.00~0.10
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)
        self.name = "AgentPPO"

    def explore_env(self, env, horizon_len: int) -> [Tensor]:
        states = torch.zeros((horizon_len, self.state_dim), dtype=torch.float32).to(self.device)
        actions = torch.zeros((horizon_len, self.action_dim), dtype=torch.float32).to(self.device)
        logprobs = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        rewards = torch.zeros(horizon_len, dtype=torch.float32).to(self.device)
        dones = torch.zeros(horizon_len, dtype=torch.bool).to(self.device)

        ary_state = self.last_state
        get_action = self.act.get_action
        convert = self.act.convert_action_for_env
        for i in range(horizon_len):
            state = torch.as_tensor(np.sum(ary_state, axis=(1, 2)), dtype=torch.float32, device=self.device)
            # print("horizon state ", i, " is ", state)
            action, logprob = [t for t in get_action(state)]
            # print("horizon action ", i, " is ", action)
            # action, logprob = [t.squeeze(0) for t in get_action(state.unsqueeze(0))[:2]]

            ary_action = np.around(((convert(action) + 1) * 3).detach().cpu().numpy(), 2)
            # print("horizon true action ", i, " is ", ary_action)
            ary_state, reward, done, info, avg_time, finished_count = env.step(ary_action)
            if done:
                ary_state = env.reset()

            states[i] = state
            actions[i] = action
            logprobs[i] = logprob
            rewards[i] = finished_count
            dones[i] = done

        self.last_state = ary_state
        rewards = (rewards * self.reward_scale).unsqueeze(1)
        undones = (1 - dones.type(torch.float32)).unsqueeze(1)
        return states, actions, logprobs, rewards, undones

    def update_net(self, buffer) -> [float]:
        with torch.no_grad():
            states, actions, logprobs, rewards, undones = buffer
            buffer_size = states.shape[0]

            '''get advantages reward_sums'''
            bs = 2 ** 10  # set a smaller 'batch_size' when out of GPU memory.
            values = [self.cri(states[i:i + bs]) for i in range(0, buffer_size, bs)]
            values = torch.cat(values, dim=0).squeeze(1)  # values.shape == (buffer_size, )

            advantages = self.get_advantages(rewards, undones, values)  # advantages.shape == (buffer_size, )
            reward_sums = advantages + values  # reward_sums.shape == (buffer_size, )
            del rewards, undones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-5)
        assert logprobs.shape == advantages.shape == reward_sums.shape == (buffer_size,)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer_size * self.repeat_times / self.batch_size)
        assert update_times >= 1
        for _ in range(update_times):
            indices = torch.randint(buffer_size, size=(self.batch_size,), requires_grad=False)
            state = states[indices]
            action = actions[indices]
            logprob = logprobs[indices]
            advantage = advantages[indices]
            reward_sum = reward_sums[indices]

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()
        a_std_log = getattr(self.act, 'a_std_log', torch.zeros(1)).mean()
        return obj_critics / update_times, obj_actors / update_times, a_std_log.item()

    def get_advantages(self, rewards: Tensor, undones: Tensor, values: Tensor) -> Tensor:
        advantages = torch.empty_like(values)  # advantage value

        masks = undones * self.gamma
        horizon_len = rewards.shape[0]

        next_state = torch.tensor(np.sum(self.last_state, axis=(1, 2)), dtype=torch.float32).to(self.device)
        # next_value = self.cri(next_state.unsqueeze(0)).detach().squeeze(1).squeeze(0)
        next_value = self.cri(next_state).detach()

        advantage = 0  # last_gae_lambda
        for t in range(horizon_len - 1, -1, -1):
            delta = rewards[t] + masks[t] * next_value - values[t]
            advantages[t] = advantage = delta + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]
        return advantages


def train_agent(args: Config):
    args.init_before_training()

    env = args.env_class
    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)
    agent.last_state = env.reset()  # (edge_num, edge_num. ddl_num)

    evaluator = Evaluator(eval_env=env,
                          eval_per_step=args.eval_per_step,
                          eval_times=args.eval_times,
                          cwd=args.cwd)

    torch.set_grad_enabled(False)
    while True:  # start training
        #env.reset()
        buffer_items = agent.explore_env(env, args.horizon_len)
        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer_items)
        torch.set_grad_enabled(False)

        evaluator.evaluate_and_save(agent.act, args.horizon_len, logging_tuple)
        if (evaluator.total_step > args.break_step) or os.path.exists(f"{args.cwd}/stop"):
            torch.save(agent.act.state_dict(), 'pkls/actor_parameters_'+today_time+'.pt')
            torch.save(agent.cri.state_dict(), 'pkls/critic_parameters_'+today_time+'.pt')
            logging.warning("The training is finished")
            break  # stop training when reach `break_step` or `mkdir cwd/stop`


class Evaluator:
    def __init__(self, eval_env, eval_per_step: int = 1e4, eval_times: int = 8, cwd: str = '.'):
        self.cwd = cwd
        self.env_eval = eval_env
        self.eval_step = 0
        self.total_step = 0
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
        avg_r = rewards_steps_ary[:, 0].mean()  # average of cumulative rewards
        std_r = rewards_steps_ary[:, 0].std()  # std of cumulative rewards
        avg_s = rewards_steps_ary[:, 1].mean()  # average of steps in an episode

        used_time = time.time() - self.start_time
        self.recorder.append((self.total_step, used_time, avg_r))

        # print(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
        #       f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
        #       f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")
        logging.warning(f"| {self.total_step:8.2e}  {used_time:8.0f}  "
                        f"| {avg_r:8.2f}  {std_r:6.2f}  {avg_s:6.0f}  "
                        f"| {logging_tuple[0]:8.2f}  {logging_tuple[1]:8.2f}")


def get_rewards_and_steps(env, actor, if_render: bool = False) -> (float, int):  # cumulative_rewards and episode_steps
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(10000):
        tensor_state = torch.as_tensor(np.sum(state, axis=(1,2)), dtype=torch.float32, device=device)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()  # not need detach(), because using torch.no_grad() outside
        action = np.expand_dims(action, axis=0)
        state, reward, done, info, avg_time, finished_count = env.step(action)
        cumulative_returns += finished_count

        if if_render:
            env.render()
        if done:
            state = env.reset()
            break
    return cumulative_returns, episode_steps + 1


def train_ppo():
    agent_class = AgentPPO
    env_class = CityFlowEnvM()
    env_args = {
        'env_name': 'cityflow',  # Apply torque on the free end to swing a pendulum into an upright position
        'state_dim': env_class.edges_num,
        # the x-y coordinates of the pendulum's free end and its angular velocity.
        'action_dim': env_class.edges_num,  # the torque applied to free end of the pendulum
        'if_discrete': False  # continuous action space, symbols → direction, value → force
    }
    print(env_class.edges_num)

    # get_gym_env_args(env=CityFlowEnvM(), if_print=True)  # return env_args

    args = Config(agent_class, env_class, env_args)  # see `config.py Arguments()` for hyperparameter explanation
    args.break_step = int(4e5)  # break training if 'total_step > break_step'
    args.net_dims = (64, 32)  # the middle layer dimension of MultiLayer Perceptron
    args.gamma = 0.97  # discount factor of future rewards
    args.repeat_times = 16  # repeatedly update network using ReplayBuffer to keep critic's loss small

    train_agent(args)


if __name__ == '__main__':
    train_ppo()
