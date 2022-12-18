#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pricing_project 
@File    ：env_ctrl.py
@Author  ：xxuanzhu
@Date    ：2022/12/5 11:58 
@Purpose :
'''

import json
import sys

import client.metaFlow as cityflow
import os
from time import time
import numpy as np

from agents.attention import FullConnection

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath[0])
sys.path.append(rootPath)


def parse_roadnet(roadnetFile):
    """
    解析 roadnet文件，生成 lane_phase_info_dict, road_info_dict, vertices（road）
    :param roadnetFile:
    :return:
    """
    roadnetData = json.load(open(roadnetFile, 'r'))

    lane_phase_info_dict = {}
    road_info_dict = {}

    # process each intersection
    for intersection in roadnetData["intersections"]:
        lane_phase_info_dict[intersection['id']] = {"start_lane": [],
                                                    "end_lane": [],
                                                    "lane_mapping": {},
                                                    "has_light": intersection['virtual']}

        # 获得当前intersection的roadlinks的startRoad 和 endRoad
        road_links = intersection["roadLinks"]

        start_lane = []
        end_lane = []
        roadLink_lane_pair = {}

        rik = 0
        for r in road_links:
            if r['type'] != 'turn_right':
                roadLink_lane_pair[rik] = []
                rik += 1

        rik = 0
        for ri in range(len(road_links)):
            road_link = road_links[ri]
            sl = road_link['startRoad']
            el = road_link['endRoad']
            start_lane.append(sl)
            end_lane.append(el)

            if road_link['type'] != 'turn_right':
                roadLink_lane_pair[rik].append([sl, el])

            if road_link['type'] != 'turn_right':
                roadLink_lane_pair[rik].append(road_link['direction'])
                rik += 1

        lane_phase_info_dict[intersection['id']]['start_lane'] = sorted(list(set(start_lane)))
        lane_phase_info_dict[intersection['id']]['end_lane'] = sorted(list(set(end_lane)))
        lane_phase_info_dict[intersection['id']]['lane_mapping'] = roadLink_lane_pair
    vertices = []
    for road in roadnetData["roads"]:
        road_info_dict[road['id']] = {"end_intersection": road["endIntersection"],
                                      "link_roads": []}
        link_roads = lane_phase_info_dict[road["endIntersection"]]["end_lane"]
        road_info_dict[road["id"]]["link_roads"] = link_roads
        road_info_dict[road['id']]['delay_index'] = 0.0
        vertices.append(
            VertexRoad(road['id'], road['startIntersection'], road['endIntersection'], link_roads)
        )

    return lane_phase_info_dict, road_info_dict, vertices


class VertexRoad:
    def __init__(self,
                 id,
                 startIntersection,
                 endIntersection,
                 adj_roads):
        self.road_id = id
        self.road_startIntersection = startIntersection
        self.road_endIntersection = endIntersection
        self.road_features = {"distance": 0.0,
                              "num_of_lights": 0,
                              "estimate_vehicles": 0}
        self.adj_roads = adj_roads
        self.delay_index = 0.0

    def set_id(self, id):
        self.road_id = id

    def get_id(self):
        return self.road_id

    def set_start(self, startIntersection):
        self.road_startIntersection = startIntersection

    def get_start(self):
        return self.road_startIntersection

    def set_end(self, endIntersection):
        self.road_endIntersection = endIntersection

    def get_end(self):
        return self.road_endIntersection

    def print(self):
        print("id:", self.road_id, "   start:", self.road_startIntersection, "   end:", self.road_endIntersection)


class CTRL_ENV(object):
    def __init__(self,
                 path_to_log='result',
                 thread_num=4,
                 cityflow_config_file='data/cityflow.config',
                 control_time=3 * 60,  # 3h,180min
                 step_time=30):  # 30min, 共6个step
        self.env_name = 'CTRL_ENV'
        self.eng = cityflow.Engine(cityflow_config_file, thread_num=thread_num)

        configData = json.load(open(cityflow_config_file))  # load cityflow config
        roadnetFile_path = configData['dir'] + configData['roadnetFile']  # get roadnet path
        flowFile_path = configData['dir'] + configData['flowFile']  # get flow path

        # store the file path
        self.roadnetFile_path = roadnetFile_path
        self.flowFile_path = flowFile_path
        self.path_to_log = path_to_log  # the dir to store result

        # parse the roadnet file, get the intersection and road data
        intersection_Data, road_Data, vertices = parse_roadnet(roadnetFile_path)  # analyze roadnet
        self.road_Data = road_Data
        self.intersection_Data = intersection_Data
        self.vertices = vertices

        # get intersection id and road id
        self.intersections_id = list(intersection_Data.keys())  # the id of intersections
        self.roads_id = list(road_Data.keys())  # the id of roads
        self.lane_vehicleId_list = list(self.get_lane_vehicles().keys())  # lane id: [vehicles1 id, vehicle2 id...]

        # nn dim
        self.vertices_num = len(self.roads_id)  # the num of roads
        self.edges_num = len(self.intersections_id)  # the num of intersections

        # time parameters
        self.control_time = control_time  # the range of control time
        self.step_time = step_time  # each step time

        # generate the alternative routes for each OD
       # flowData = json.load(open(flowFile_path))  # get flow data
        self.origin_set = set()
        self.dest_set = set()
        self.all_path = {}

        # 以下二选一
        # one，先得到OD set， 求每个O到每个D的候选路径
        # self.origin_set, self.dest_set = self.get_OD_set(flowData)
        # self.all_path = self.generate_topk_paths()

        # two，只求flow文件中出现的OD的 top 5 distance路径
        # self.all_path = self.get_OD_set_and_paths(flowData)

        # 得到每个OD的三条候选路径
        self.alternative_3_routes = {}
        # self.alternative_3_routes = self.generate_alternative_3_routes()  # generate 3 routes for each OD

        # road view
        self.road_features = None  # the state to enter self attention

        # route view
        self.route_state = None
        self.action = None


        self.fc_NN = FullConnection()


    def get_lane_vehicles(self):
        """
        Get vehicle ids on each lane.
        Return a dict with lane id as key and list of vehicle id as value.
        :return:
        """
        return self.eng.get_lane_vehicles_id()

    def generate_alternative_3_routes(self):
        """
        获得所有OD 的3条备选路径
        :return:
        """
        alternative_3_routes = {}
        for road in self.all_path:
            alternative_3_routes[road] = {}
            intersections = self.all_path[road]
            for intersection in intersections:
                alternative_3_routes[road][intersection] = {
                    'min_distance': self.find_shortest(self.all_path[road][intersection]),
                    'min_lights': self.find_less_light(self.all_path[road][intersection]),
                    'min_vehicles': self.find_less_vehicle(self.all_path[road][intersection])
                }
        return alternative_3_routes

        # for road in self.roads_id:
        #     vi = self.road_Data[road]['end_intersection']
        #     # 以road为起点的三条路径
        #     alternative_3_routes[road] = {
        #         "road": road,
        #         "terminal": [],
        #         "routes": {}
        #     }
        #     for intersection in self.intersections_id:
        #         paths = self.get_all_paths(road, intersection)
        #         # 根据距离获取前k条路径
        #         new_paths = sorted(paths, key=lambda x: self.cal_distance(x))
        #         alternative_k_routes = []
        #         k = min(10, len(new_paths))
        #         for i in range(k):
        #             alternative_k_routes.append(new_paths[i])
        #
        #         # final_paths = self.get_final_3_pahts(paths)
        #         alternative_3_routes[road]['terminal'].append(intersection)
        #         alternative_3_routes[road]['routes']['min_dis'] = final_paths[0]
        #         alternative_3_routes[road]['routes']['min_lights'] = final_paths[1]
        #         alternative_3_routes[road]['routes']['min_vehicles'] = final_paths[2]
        return alternative_3_routes



    def get_all_paths(self, road_i, vj, path=[], intersection=[]):
        """
        获得从vi出发到vj的所有可能路径(顶点）
        :param road_i:
        :param vj:
        :param path:
        :param intersection:
        :return:
        """
        path = path + [road_i]
        vi = self.road_Data[road_i]["end_intersection"]
        intersection = intersection + [vi]
        if vi == vj:
            return [path]
        paths = []

        for road in self.road_Data[road_i]['link_roads']:
            if self.road_Data[road]["end_intersection"] not in intersection:
                new_paths = self.get_all_paths(road, vj, path, intersection)
                for new_path in new_paths:
                    paths.append(new_path)
        return paths



    def find_shortest(self, paths):
        """
        获取距离最短的路径
        :param paths:
        :return:
        """
        min_dis = float('inf')
        min_dis_path = None
        for path in paths:
            dis = self.cal_distance(path)
            if dis < min_dis:
                min_dis = dis
                min_dis_path = path

        return min_dis_path

    def find_less_light(self, paths):
        """
        获取lights最少的路径
        :param paths:
        :return:
        """
        min_light_num = float('inf')
        min_light_path = None
        for path in paths:
            count = 0
            for road in path:
                if self.intersection_Data[self.road_Data[road]['end_intersection']]['has_light'] is True:
                    count += 1
            if count < min_light_num:
                min_light_num = count
                min_light_path = path

        return min_light_path

    def find_less_vehicle(self, paths):
        """
        获取预估车辆数最少的路径
        :param paths:
        :return:
        """
        min_vehicle_num = float('inf')
        min_vehicle_path = None

        vehicle_num_count_dict = self.eng.get_lane_vehicles_count()
        process_count_dict = self.get_road_count(vehicle_num_count_dict)

        for path in paths:
            vehicle_num = 0
            for road in path:
                vehicle_num += process_count_dict[road]
            if vehicle_num < min_vehicle_num:
                min_vehicle_num = vehicle_num
                min_vehicle_path = path
        return min_vehicle_path

    def get_road_count(self, vehicle_num_count_dict):
        """
        获取每条road上的正在行驶的车辆数量
        :param vehicle_num_count_dict:
        :return:
        """
        process_count_dict = {}
        for road in vehicle_num_count_dict:
            key = road[:10]
            if key in process_count_dict:
                process_count_dict[key] = process_count_dict[key] + vehicle_num_count_dict[road]
            else:
                process_count_dict[key] = vehicle_num_count_dict[road]
        return process_count_dict

    def next_step(self):
        self.eng.next_step()
        new_enter_vehicles_id = self.eng.get_new_enter_vehicles_id()
        for vehicle in new_enter_vehicles_id:
            path = self.eng.get_vehicle_info(vehicleId)['route']
            origin = path[0]
            dest = path[-1]
            self.eng.change_vehicle_routeList(vehicleId, best_path)

    def reset(self):
        self.eng.reset()
        self.global_delay_index = 0.0
        self.create_state()
        self.create_action()
        self.t = 0
        return self.state

    # TODO
    def create_state(self):
        # 运行10个step，获取初始状态
        for i in range(10):
            self.next_step()
        # 得到每个road的observation
        self.road_features = {}
        for road in self.roads_id:
            self.road_features[road] = 0.0
        # for road in self.alternative_3_routes:
        #     intersections = self.alternative_3_routes[road]
        #     for intersection in intersections:
        #         min_dis_path = self.alternative_3_routes[road][intersection]['min_distance']
        #         min_lights_path = self.alternative_3_routes[road][intersection]['min_lights']
        #         min_vehicles_path = self.alternative_3_routes[road][intersection]['min_vehicles']
        self.route_state = np.zeros(3, dtype=np.float64)

    # TODO
    def create_action(self):
        """
        创建action向量，三条路径的价格，shape 3
        :return:
        """
        self.action = np.zeros(3, dtype=np.float64)

    def cal_distance(self, path):
        """
        得到一条route的distance
        :param path:
        :return:
        """
        dis = 0.0
        for road in path:
            dis += self.eng.get_road_length(road)

        return dis

    # TODO
    def step(self,
             action,
             ):
        stime = time()
        next_state = None
        global_delay_index = 0.0
        is_done = False
        info = None
        s = self.road_features

        for i in range(self.step_time * 60):
            # 对刚进入路网的车辆选择路径
            self.next_step()

        # 计算delay index

        lane_vehicles = self.get_lane_vehicles()
        lane_list = self.lane_vehicleId_list

        # state 三种路径的delay数组
        # reward  =  d_global
        return state ,reward

    def get_OD_set_and_paths(self, flowData):
        """
        得到每个OD对的以路径作为标准，最短的5条路径
        :param flowData:
        :return:
        """
        all_path = {}
        for vehicle in flowData:
            path = vehicle['route']
            origin = path[0]
            dest = path[-1]
            if origin in self.origin_set and dest in self.dest_set:
                continue
            else:
                self.origin_set.add(origin)
                self.dest_set.add(dest)
                destination = 'intersection_' + dest[5:8]
                all_path[origin] = {}
                all_path[origin][destination] = []
                paths = self.get_all_paths(origin, destination)
                new_paths = sorted(paths, key=lambda x: self.cal_distance(x))
                k = min(5, len(new_paths))
                for i in range(k):
                    all_path[origin][destination].append(new_paths[i])
        return all_path

    def get_OD_set(self, flowData):
        """
        根据flowData得到 origin set 和 destination set
        :param flowData:
        :return:
        """
        origin_set = set()
        dest_set = set()
        for vehicle in flowData:
            path = vehicle['route']
            origin = path[0]
            dest = path[-1]
            origin_set.add(origin)
            dest_set.add(dest)
        return origin_set, dest_set

    def generate_topk_paths(self):
        """
        得到origin set中每个origin 到 destination set中每个destination的 所有路径
        :return:
        """
        all_paths = {}
        for origin in self.origin_set:
            all_paths[origin] = {}
            for dest in self.dest_set:
                destination = 'intersection_' + dest[5:8]
                all_paths[origin][destination] = []
                paths = self.get_all_paths(origin, destination)
                new_paths = sorted(paths, key=lambda x: self.cal_distance(x))
                k = min(5, len(new_paths))
                for i in range(k):
                    all_paths[origin][destination].append(new_paths[i])
        return all_paths

    def get_final_3_paths(self, paths):
        """
        :param paths:
        :return:  # 获取每个所有的3条候选路径
        """
        # 根据距离获取前k条路径
        new_paths = sorted(paths, key=lambda x: self.cal_distance(x))
        alternative_k_routes = []
        k = min(10, len(new_paths))
        for i in range(k):
            alternative_k_routes.append(new_paths[i])

        return_paths = []
        if len(alternative_k_routes) > 3:
            shortest_path = self.find_shortest(alternative_k_routes)  # shortest distance route
            less_light_path = self.find_less_light(alternative_k_routes)  # less lights route
            less_estimate_vehicles_path = self.find_less_vehicle(
                alternative_k_routes)  # less estimate vehicles nums route
            return_paths.append(shortest_path)
            return_paths.append(less_light_path)
            return_paths.append(less_estimate_vehicles_path)
        else:
            for path in alternative_k_routes:
                return_paths.append(path)

        return return_paths

if __name__ == '__main__':
    cityflow_config_file = 'data/cityflow.config'
    eng = cityflow.Engine(cityflow_config_file, thread_num=4)
    for i in range(600):
        eng.next_step()
        # data = env.eng.get_vehicles_id()
        # for v in data:
        #     info = env.eng.get_vehicle_info(v)
        #     y = 1
        # x = 1
