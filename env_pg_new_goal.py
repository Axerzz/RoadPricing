#!/usr/bin/env python 
# -*- coding:utf-8 -*-

'''
此类实现功能：
执行动作后的状态转换，奖励的计算，以及判断是否到达终点
'''

import numpy as np
import random
import math
from new_goal.utils import Graph, Path, Edge
from scipy import stats

FIXED_ROAD_LENGTH = 3  # km


# BPR模型的参数


class TrafficEnvironment:
    def __init__(self,
                 road_network: Graph,
                 min_action=0,
                 max_action=6,
                 # action_limit=6,
                 control_time=6,  # 单位小时
                 period_time=10,  # 单位分钟
                 free_flow_speed=5,
                 capacity_of_road=200,
                 w_=0.5,
                 w=0.5,

                 ):
        self.road_network = road_network
        self.free_flow_speed = free_flow_speed  # free_flow_speed，不堵车时的travel speed
        self.capacity_of_road = capacity_of_road  # 每条道路的容量，假设只有一个车道
        self.zones_num = self.road_network.vertex_num()  # 目的地总数
        self.edges_num = self.road_network.get_edge_num()  # 道路总数
        self.edges = self.road_network.generate_edges()  # 得到道路数组
        self.control_time = control_time * 60  # 总的需要进行流量控制的时长，单位min
        self.period_time = period_time  # 每个时间间隔的时长，单位min
        self.deadline_num = int(self.control_time / self.period_time)
        self.w_ = w_  # sensitivity to travel cost
        self.w = w  # value of time
        self.A = 0.15
        self.B = 4
        self.t = 0
        self.state_matrix = None
        self.action_vector = None
        # self.reward = np.zeros((self.edges_num, self.zones_num), dtype=int)
        self.low_bound_action = min_action
        self.upper_bound_action = max_action

    def reset(self):
        self.create_state_matrix()
        self.create_action_vector()
        self.t = 1
        return self.state_matrix

    # 初始化状态矩阵
    def create_state_matrix(self):
        # zones,  # 目的地数目
        # edges,  # 总的路数
        self.state_matrix = np.random.randint(int(0.5 * self.capacity_of_road / 4),
                                              int(0.7 * self.capacity_of_road / 4), (self.edges_num, self.zones_num, self.deadline_num))

        s = self.state_matrix
        sum_s = s.sum(axis=1)


    # 创建初始行为向量
    def create_action_vector(self):
        self.action_vector = np.random.randint(0, 6, self.edges_num)

    # 生成随机VOT
    # def create_vot(self, d):
    #     mu, sigma = 0,1
    #     norm_lower = np.log(0.000000001)
    #     norm_upper = np.log(1)
    #     X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
    #     norm_data = X.rvs(data_num)
    #     log_data = np.exp(norm_data)

    def travel_time(self,
                    road_e: Edge,  # 此变量用来填充state_matrix的一维坐标，Edge；{id，start，end}
                    d  # deadline
                    ):
        # state_matrix：状态矩阵，state_matrix[i][j]代表目的地是j的路i上的车辆数
        # 输出时段t，road_e上的travel time，是个vector，代表到不同目的地
        # print(self.state_matrix[road_e.id])
        # print(self.free_flow_speed * (1 + A * (self.state_matrix[road_e.id] / self.capacity_of_road) ** B))
        # print(sum(self.state_matrix[road_e.id]))
        s = self.state_matrix
        sum_s = s.sum(axis=1)
        return self.free_flow_speed * (
                    1 + self.A * (sum_s[road_e.id, d] / self.capacity_of_road) ** self.B)

    # 从i到j的某条路径的cost
    def travel_cost(self,
                    path: Path,  # 到目的地j的某条路径,path传入的路段的id，不是区域顶点
                    d, # deadline
                    ):
        sum = 0
        # destination_index = path.get_dest()
        for e in path.get_path():
            sum += self.action_vector[0, e] + 1/(d+1) * self.travel_time(self.edges[e], d)
        return sum

    # 某条路径的traffic demand
    def traffic_demand(self, vi, vj, one_path: Path, d):
        # 计算单独某条
        # single_path = math.exp(self.w_ * self.travel_cost(one_path))
        single_path = round(np.exp((-self.w_) * self.travel_cost(one_path, d)), 8)

        # 计算所有路径
        all_Paths = self.road_network.return_all_path_roads(vi, vj, path=[])
        sum_all_Paths = 0
        for row in all_Paths:
            p = Path(vi, vj, row)
            # sum_all_Paths += math.exp(self.w_ * self.travel_cost(p))
            sum_all_Paths += round(np.exp((-self.w_) * self.travel_cost(p, d)), 8)

        if sum_all_Paths == 0:
            resu = 0
        else:
            resu = single_path / sum_all_Paths
        return resu

    # 计算t时段内 离开road_e，目的地是vj的车辆数
    def out_road(self,
                 road_e: Edge,
                 vj,
                 d,
                 ):
        s = self.state_matrix
        return int(s[road_e.id, vj, d] * self.period_time / self.travel_time(road_e, d))

    # 计算secondary demand
    def secondery_demand(self,
                         vi,
                         vj,
                         d,
                         ):
        out_edges = self.road_network.in_edges(vi)  # 获取vi的入边
        sum = 0
        for i in range(len(out_edges)):
            sum += self.out_road(out_edges[i], vj, d)
        return sum

    # 计算primary_demand
    # 每条i，j都可以这样计算
    def primary_demand(self):
        return int(random.randint(80, 120) * np.exp(-np.power(self.t - 18, 2) / (2 * np.power(30, 2))))
        # return int(random.randint(24,36) * np.exp(-np.power(self.t - 60, 2) / (2 * np.power(30, 2))))

    # 计算t时段内 进入road_e的车辆数
    def in_road(self,
                road_e: Edge,
                vj,
                d,
                ):
        # 以edge的起点
        start = road_e.start
        primary = self.primary_demand()
        secondary = self.secondery_demand(start, vj, d)
        # 所有以road_e的起点为终点的路径
        # 从vi到vi的所有路径
        all_Paths = self.road_network.return_all_path_roads(start, vj)
        in_road_cars = 0
        # 找到包含road_e的path
        for i in all_Paths:
            if road_e.id in i:
                in_road_cars += (primary + secondary) * self.traffic_demand(start, vj, Path(start, vj, i), d)
        if math.isnan(in_road_cars):
            in_road_cars = 0
        return int(in_road_cars)

    #
    def step(self,
             action_vector,  # 传入新的收费值
             ):
        self.action_vector = action_vector
        next_state_matrix = np.zeros((self.edges_num, self.zones_num, self.deadline_num), dtype=int)
        rewards = 0
        is_done = False
        info = None
        s = self.state_matrix

        # 计算下个状态矩阵
        for e in range(self.edges_num):
            for j in range(self.zones_num):
                for z in range(self.deadline_num):
                    # t时段在road_e上的车辆数
                    now_on_road = self.state_matrix[e, j, z]
                    # t时段从road_e上离开的车辆数
                    off_road = self.out_road(self.edges[e], j, z)
                    # t时段进入road_e的车辆数
                    in_road = self. in_road(self.edges[e], j, z)
                    next_state_matrix[e, j, z] = now_on_road - off_road + in_road
                    if next_state_matrix[e, j, z] < 0:
                        next_state_matrix[e, j, z] = 0
        self.state_matrix = next_state_matrix


        sum_s = s.sum(axis=1)

        # 计算奖励
        for e in self.edges:
            end_point = e.end
            rewards += (self.state_matrix[e.id, end_point, self.t] * self.period_time) / (self.free_flow_speed * (
                        1 + self.A * (sum_s[e.id, self.t] / self.capacity_of_road) ** self.B))
        rewards = int(rewards)

        self.t += 1
        if self.t == self.control_time / self.period_time:
            is_done = True

        return self.state_matrix, rewards, is_done, info
