import torch.nn as nn
from utility import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import math, os
import time
os.environ["OMP_NUM_THREADS"] = "1"
from config import *
import matplotlib
matplotlib.use("Agg")

config = Config()

UPDATE_GLOBAL_ITER = 5
GAMMA = 1
MAX_EP = 1000
MAX_EP_STEP = 36

env = config.ENV
N_S = config.observation_input
N_A = config.action_output


class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = nn.Linear(s_dim, 200)  # [10,1444] -> [1444,200]
        self.mu = nn.Linear(200, a_dim)
        self.sigma = nn.Linear(200, a_dim)
        self.c1 = nn.Linear(s_dim, 100)
        self.v = nn.Linear(100, 1)
        set_init([self.a1, self.mu, self.sigma, self.c1, self.v])
        self.distribution = torch.distributions.Beta

    def forward(self, x):
        a1 = F.relu6(self.a1(x))  # [10,200]
        lamda = 2 * torch.tanh(self.mu(a1))  # [10,1]
        epislon = F.softplus(self.sigma(a1)) + 0.001  # avoid 0,[10,1]
        c1 = F.relu6(self.c1(x))  # [10,100]
        values = self.v(c1)  # [10,1]
        return lamda, epislon, values

    def choose_action(self, s):
        self.training = False
        lamda, epislon, _ = self.forward(s)  # [10,1],[10,1],[10,1]
        m = self.distribution(lamda.data, epislon.data)  # mu:[10,1],sigma:[10,1]
        return m.sample().cpu().numpy() * 6

    def loss_func(self, s, a, v_t):
        self.train()
        lamda, epislon, values = self.forward(s)
        td = v_t - values
        # print(values)
        c_loss = td.pow(2)

        m = self.distribution(lamda, epislon)
        log_prob = m.log_prob(a)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(epislon) # exploration
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()
        #print(total_loss)
        return total_loss


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep = global_ep
        self.g_ep_r = global_ep_r
        self.res_queue = res_queue  # 全局的ep，ep_r, res_queue
        self.gnet = gnet.to(config.device)
        self.opt = opt
        self.lnet = Net(N_S, N_A).to(config.device)  # local network
        if (os.path.exists(
            'parameters/A3C_parameters.pkl')): # True/False
            self.lnet.load_state_dict(
                        torch.load('parameters/DyETC_parameters.pkl'))
        self.env = config.ENV  # 环境

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            time_start = time.time()
            s = self.env.reset()  # 初始状态
            buffer_s, buffer_a, buffer_r = [], [], []  # 存储state，action，reward
            ep_r = 0.  # expected reward
            for t in range(MAX_EP_STEP):
                with torch.no_grad():
                    a = self.lnet.choose_action(v_wrap(s))
                    # print(a.clip(0, 6))
                s_, r, done, _ = self.env.step(a.clip(0, 6))  # 与环境交互
                if t == MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r / 1e4)  # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # 每 UPDATE_GLOBAL_ITER 步 或者回合完了, 进行 sync 操作
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        time_end = time.time()
                        print('totally time cost', time_end - time_start)
                        torch.save(self.gnet.state_dict(), 'parameters/DyETC_parameters.pkl')
                        break
                s = s_
                total_step += 1

        self.res_queue.put(None)


if __name__ == "__main__":

    import multiprocessing as mp

    mp.set_start_method('spawn')
    gnet = Net(N_S, N_A).to(config.device)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-10, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()  # 全局的期望xx，期望奖励，结果队列

    # parallel training
    # workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(10)]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt

    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.savefig('res_DyETC.jpg')
