# import cityflow
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath[0])
sys.path.append(rootPath)

import client.metaFlow as cityflow
import pandas as pd
import os
from time import time
import math
import numpy as np
import json
import itertools
import random
from new_goal.utils import Graph, Path, Edge
from new_goal.utils import *
from scipy import stats


class CityFlowEnvM(object):
    '''
    cityflow environment
    '''

    def __init__(self,
                 # dic_traffic_env_conf,
                 num_step=2000,
                 path_to_log='result',
                 thread_num=4,
                 cityflow_config_file='data/cityflow.config',
                 min_action=0,
                 max_action=6,
                 # action_limit=6,
                 control_time=1,  # 单位小时
                 period_time=10,  # 单位分钟
                 free_flow_speed=5,
                 capacity_of_road=200,
                 w_=0.5,
                 w=0.5,

                 ):
        self.eng = cityflow.Engine(cityflow_config_file, thread_num=thread_num)
        self.car_ddl = {}

        configData = json.load(open(cityflow_config_file))
        roadnetFile_path = configData['dir'] + configData['roadnetFile']
        flowFile_path = configData['dir'] + configData['flowFile']

        intersection_Data, road_Data = parse_roadnet(roadnetFile_path)

        self.road_Data = road_Data

        self.intersection_id = list(intersection_Data.keys())

        self.lane_list = list(self.get_lane_vehicles().keys())

        self.road_id = list(road_Data.keys())
        # print(intersection_id)

        self.topk_routes = {}

        self.roadnetFile = roadnetFile_path
        self.flowFile = flowFile_path

        self.zones_num = len(self.intersection_id)  # 目的地总数
        self.edges_num = len(self.road_id)  # 道路总数

        self.edges = self.generate_edges(self.roadnetFile)

        self.num_step = num_step
        self.path_to_log = path_to_log
        self.free_flow_speed = free_flow_speed  # free_flow_speed，不堵车时的travel speed
        self.capacity_of_road = capacity_of_road  # 每条道路的容量，假设只有一个车道
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

        self.genernate_topk_routes()

        flowData = json.load(open(flowFile_path))
        i = 0
        for vehicle in flowData:
            flow_id = "flow_" + str(i) + "_0"
            self.car_ddl[flow_id] = np.random.randint(0, self.deadline_num)
            i = i + 1

        # config_dict = {
        #     "interval": self.dic_traffic_env_conf["INTERVAL"],
        #     "seed": 0,
        #     "laneChange": False,
        #     "dir": "",
        #     "roadnetFile": os.path.join(self.path_to_log, self.dic_traffic_env_conf['ROADNET_FILE']),
        #     "flowFile": os.path.join(self.path_to_log, self.dic_traffic_env_conf["FLOW_FILE"]),
        #     "rlTrafficLight": self.dic_traffic_env_conf["RLTRAFFICLIGHT"],
        #     "saveReplay": self.dic_traffic_env_conf["SAVEREPLAY"],
        #     "roadnetLogFile": "frontend/roadnet.json",
        #     "replayLogFile": "frontend/replay.txt"
        # }
        # config_path = os.path.join(path_to_log, "cityflow_config")
        # with open(config_path, "w") as f:
        #     config_obj = json.dump(config_dict, f)
        #     print("dump cityflow config")
        #     print(config_path)
        # self.eng = cityflow.Engine(config_path, thread_num=thread_num)

    def genernate_topk_routes(self):
        for road in self.road_id:
            vi = self.road_Data[road]["end_intersection"]
            self.topk_routes[road] = {
                "road": road,
                "terminal": [],
                "routes": []

            }
            for intersection in self.intersection_id:
                    paths = self.get_Path(road, intersection)
                    new_paths = sorted(paths, key=lambda x: len(x))
                    top_k_routes = []
                    m = min(3, len(new_paths))
                    for i in range(m):
                        top_k_routes.append(new_paths[i])
                    self.topk_routes[road]["terminal"].append(intersection)
                    self.topk_routes[road]["routes"].append(top_k_routes)
                # self.topk_routes[road]["route_list"].append()

    def generate_edges(self,
                       roadnet_file_path,
                       ):
        roadnetData = json.load(open(roadnet_file_path, 'r'))
        edges = []
        for i in range(len(roadnetData['roads'])):
            edges.append(
                Edge(roadnetData['roads'][i]['id'], roadnetData['roads'][i]['startIntersection'],
                     roadnetData['roads'][i]['endIntersection']))
        # for i in range(len(roadnetData['roads'])):
        #     edges[i].print()
        return edges

    def bulk_log(self):
        # self.eng.print_log(os.path.join(self.path_to_log, "replay.txt"))
        self.eng.set_replay_file((os.path.join(self.path_to_log, "replay.txt")))

    def get_vehicles(self):
        return self.eng.get_vehicles()

    def get_current_time(self):
        return self.eng.get_current_time()

    def get_average_travel_time(self):
        return self.eng.get_average_travel_time()

    def get_finished_vehicle_count(self):
        return self.eng.get_finished_vehicle_count()

    def set_vehicle_route(self, vehicleId, anchorId):
        return self.eng.set_vehicle_route(vehicleId, anchorId)

    def get_vehicle_info(self, ID):
        return self.eng.get_vehicle_info(ID)

    def get_lane_vehicles(self):
        return self.eng.get_lane_vehicles()

    def get_vehicle_des(self, ID):
        des_road = self.eng.get_vehicle_destination(ID)
        des_intersection = self.road_Data[des_road]["end_intersection"]
        return des_intersection

    def next_step(self):
        self.eng.next_step()

    def close(self):
        self.eng.close()

    def reset(self):
        self.eng.reset()
        self.create_state_matrix()
        self.create_action_vector()
        self.t = 1
        return self.state_matrix

    # 初始化状态矩阵
    def create_state_matrix(self):
        # zones,  # 目的地数目
        # edges,  # 总的路数
        self.state_matrix = np.zeros((self.edges_num, self.zones_num, self.deadline_num), dtype=int)
        self.next_step()
        lane_vehicles = self.get_lane_vehicles()

        lane_list = self.lane_list

        for lane in lane_list:
            for vehicle in lane_vehicles[lane]:
                vehicle_des = self.get_vehicle_des(vehicle)

                i = self.road_id.index(lane[:-2])
                j = self.intersection_id.index(vehicle_des)
                ddl = self.car_ddl[vehicle]
                self.state_matrix[i][j][ddl] += 1

    # 创建初始行为向量
    def create_action_vector(self):
        self.action_vector = np.random.randint(0, 6, self.edges_num)

    def travel_time(self,
                    road_e,  # 此变量用来填充state_matrix的一维坐标，Edge；{id，start，end}
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
                1 + self.A * (sum_s[self.road_id.index(road_e), d] / self.capacity_of_road) ** self.B)

    # 获得从vi出发到vj的所有可能路径(顶点）
    def get_Path(self, road_i, vj, path=[], intersection=[]):
        path = path + [road_i]
        vi = self.road_Data[road_i]["end_intersection"]
        intersection = intersection + [vi]
        if vi == vj:
            return [path]

        paths = []
        for road in self.road_Data[road_i]['link_roads']:
            if self.road_Data[road]["end_intersection"] not in intersection:
                newpaths = self.get_Path(road, vj, path, intersection)
                for newpath in newpaths:
                    paths.append(newpath)

        return paths

    # 从i到j的某条路径的cost
    def travel_cost(self,
                    path,  # 到目的地j的某条路径,path传入的路段的id，不是区域顶点
                    d,  # deadline
                    ):
        sum = 0
        # destination_index = path.get_dest()
        for e in path:
            sum += self.action_vector[0, int(self.road_id.index(e))] \
                   + 1 / (d + 1) * self.travel_time(e, d)
        return sum

    def step(self,
             action_vector,  # 传入新的收费值
             ):
        t1 = time()
        self.action_vector = action_vector
        next_state_matrix = np.zeros((self.edges_num, self.zones_num, self.deadline_num), dtype=int)
        rewards = 0
        is_done = False
        info = None
        s = self.state_matrix

        # 根据cityflow获取state-matrix矩阵
        for i in range(self.period_time * 60):
            self.next_step()

        lane_vehicles = self.get_lane_vehicles()

        lane_list = self.lane_list

        for lane in lane_list:
            for vehicle in lane_vehicles[lane]:
                vehicle_des = self.get_vehicle_des(vehicle)
                vehicle_info = self.get_vehicle_info(vehicle)
                next_route = vehicle_info['route'].split(" ")[1]

                i = self.road_id.index(lane[:-2])
                j = self.intersection_id.index(vehicle_des)

                ddl = self.car_ddl[vehicle]
                next_state_matrix[i][j][ddl] += 1

                # 修改vehicle route
                if next_route != "":
                    paths = self.topk_routes[next_route]['routes'][j]

                    # paths = self.get_Path(lane[:-2], vehicle_des)
                    best_path = []
                    min_cost = 1e9
                    for path in paths:
                        cost = self.travel_cost(path, ddl)
                        if cost < min_cost:
                            min_cost =cost
                            best_path = path
                    self.set_vehicle_route(vehicle, best_path)

        self.state_matrix = next_state_matrix
        sum_s = s.sum(axis=1)

        # 计算奖励
        for e in self.edges:
            end_point = self.intersection_id.index(e.end)
            id = self.road_id.index(e.id)
            rewards += (self.state_matrix[id, end_point, self.t] * self.period_time) / (self.free_flow_speed * (
                    1 + self.A * (sum_s[id, self.t] / self.capacity_of_road) ** self.B))
        rewards = int(rewards)

        self.t += 1
        if self.t == self.control_time / self.period_time:
            is_done = True

        simu_time = time() - t1

        avg_time = self.get_average_travel_time()

        finished_count = self.get_finished_vehicle_count()

        return self.state_matrix, rewards, is_done, info , avg_time , finished_count


if __name__ == '__main__':
    env = CityFlowEnvM()

    env.reset()
    for i in range(500):
        env.next_step()
        if i == 30:
            x = env.get_vehicle_info("flow_0_0")
            route = x['route'].split(" ")[1]
            print(x['route'])
            print(route)
            print(env.get_vehicle_info("flow_0_0"))
            env.set_vehicle_route("flow_0_0", ['road_4_1_1', 'road_4_2_0'])
        if i == 100:
            print(env.get_vehicles())
            print(env.get_vehicle_info("flow_0_0"))

    # print(env.get_current_time())
    # #
    # print(env.get_lane_vehicles())
    # #
    env.close()
