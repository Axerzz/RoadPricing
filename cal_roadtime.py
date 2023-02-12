#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：pricing_project
@File    ：test.py
@Author  ：xxuanzhu
@Date    ：2022/10/25 9:48 
@Purpose :
'''

import numpy as np
import torch
import logging
import time
import json
import client.metaFlow as cityflow

route_down = ["road_4_2_1", "road_4_3_1", "road_4_14_1", "road_4_4_1", "road_4_5_1", "road_4_6_1", "road_4_7_1", "road_4_8_1",
              "road_4_9_1", "road_4_17_1"]

tmp_down = ["road_4_14_1", "road_4_4_1", "road_4_5_1", "road_4_6_1", "road_4_7_1", "road_4_8_1", "road_4_9_1"]
tmp_up = ["road_4_14_0", "road_4_15_1", "road_4_16_2"]


# route_up = ["road_4_3_1", "road_4_14_0", "road_4_15_1", "road_4_16_2", "road_4_17_1"]


def get_road_time(env, road):
    road_record = env.get_road_record(road)
    j = 0
    total_time = 0
    for iter in road_record:
        if road_record[iter][1] > 0:
            j += 1
            total_time += road_record[iter][1] - road_record[iter][0]
    lambda_road = total_time / j
    return lambda_road


route_up = ["road_4_2_1", "road_4_3_1", "road_4_14_0", "road_4_15_1", "road_4_16_2", "road_4_17_1"]

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


def get_road_delay(env, road):
    # calculate lambda_road
    lambda_road = 0.0
    road_record = env.get_road_record(road)
    j = 0
    total_time = 0
    for iter in road_record:
        if road_record[iter][1] != -1:
            j += 1
            total_time += road_record[iter][1] - road_record[iter][0]
    if j == 0:
        return 0
    lambda_road = total_time / j
    return lambda_road


if __name__ == '__main__':
    cityflow_config_file = 'data/test.config'
    env = cityflow.Engine(cityflow_config_file, thread_num=4)
    configData = json.load(open(cityflow_config_file))  # load cityflow config
    roadnetFile_path = configData['dir'] + configData['roadnetFile']  # get roadnet path
    intersection_Data, road_Data, vertices = parse_roadnet(roadnetFile_path)  # analyze roadnet
    roads_id = list(road_Data.keys())  # the id of roads
    for i in range(3000):
        env.next_step()
    for road in roads_id:
        if road.split("_")[3]!="3":
            s = "{\"" + road + "\":"
            print(s, get_road_delay(env, road),"}")
            
    # env.next_step()
    # new_enter_vehicles_id = env.get_new_enter_vehicles_id()
    # print(env.get_vehicle_info("flow_0_0"))
    # for vehicle in new_enter_vehicles_id:
    #     print(vehicle)
    #     env.change_vehicle_routeList(vehicle, route_up)
    # print(env.get_vehicle_info("flow_0_0"))

    up_cnt = 0
    down_cnt = 0
    lane_list = list(env.get_lane_vehicles_count().keys())
    for i in range(3000):
        up_vehicle = 0
        down_vehicle = 0
        env.next_step()
        new_enter_vehicles_id = env.get_new_enter_vehicles_id()
        for vehicle in new_enter_vehicles_id:
            lane_vehicles = env.get_lane_vehicles_count()
            for lane in lane_list:
                road = lane[:-2]
                if road in tmp_up:
                    up_vehicle += lane_vehicles[lane]
                else:
                    if road in tmp_down:
                        down_vehicle += lane_vehicles[lane]
            if up_vehicle >= 1.5 * down_vehicle:
                down_cnt += 1
                env.change_vehicle_routeList(vehicle, route_down)
            else:
                up_cnt += 1
                env.change_vehicle_routeList(vehicle, route_up)
    print(down_cnt, up_cnt)
    

