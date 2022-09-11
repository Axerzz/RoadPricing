#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import json


class Edge:
    def __init__(self,
                 id,
                 start,
                 end):
        self.id = id
        self.start = start
        self.end = end

    def set_id(self, id):
        self.id = id

    def get_id(self):
        return self.id

    def set_start(self, start):
        self.start = start

    def get_start(self):
        return self.start

    def set_end(self, end):
        self.end = end

    def get_end(self):
        return self.end

    def print(self):
        print("id:", self.id, "   start:", self.start, "   end:", self.end)


class Graph:
    def __init__(self,
                 mat,
                 unconn=0):
        vnum = len(mat)
        for x in mat:
            if len(x) != vnum:
                raise ValueError("参数错误")
        self._mat = [mat[i][:] for i in range(vnum)]
        self._unconn = unconn
        self._vnum = vnum
        self.edges = []

    def generate_edges(self):
        for i in range(self.vertex_num()):
            for j in range(self.vertex_num()):
                if self._mat[i][j] != 0:
                    self.edges.append(self._mat[i][j])
        return self.edges

    def vertex_num(self):
        return self._vnum

    def _invalid(self, v):
        return v < 0 or v >= self._vnum

    # 获取边的值
    def get_edge(self, vi, vj):
        if self._invalid(vi) or self._invalid(vj):
            raise ValueError(str(vi) + "or" + str(vj) + "不是有效的顶点")
        return self._mat[vi][vj]

    # 获得一个顶点的各条入边(Edge)
    def in_edges(self, vi):
        # if self._invalid(vi):
        #     raise ValueError(str(vi) + "不是有效的顶点")
        # return self._out_edges(self._mat[vi], self._unconn)
        edges = []
        for i in range(self.vertex_num()):
            if self._mat[i][vi] != 0:
                edges.append(self._mat[i][vi])
        return edges

    # 获得一个顶点的各条出边（Edge）
    def out_edges(self, vi):
        # if self._invalid(vi):
        #     raise ValueError(str(vi) + "不是有效的顶点")
        # return self._out_edges(self._mat[vi], self._unconn)
        edges = []
        for i in range(self.vertex_num()):
            if self._mat[vi][i] != 0:
                edges.append(self._mat[vi][i])
        return edges

    # 获得一共有多少条边（有向图）
    def get_edge_num(self):
        edges_num = 0
        for i in range(self.vertex_num()):
            for j in range(self.vertex_num()):
                if self.get_edge(i, j) != 0:
                    edges_num += 1
        return edges_num

    # 获得从vi出发到vj的所有可能路径(顶点）
    def get_Path(self, vi, vj, path=[]):
        path = path + [vi]
        if vi == vj:
            return [path]

        paths = []
        for node in range(self.vertex_num()):
            if node not in path and self.get_edge(vi, node) != 0:
                newpaths = self.get_Path(node, vj, path)
                for newpath in newpaths:
                    paths.append(newpath)

        return paths

    # 获得从vi出发到vj的所有可能路径(边）
    def return_all_path_roads(self, vi, vj, path=[]):
        paths = self.get_Path(vi, vj, path)

        roads = []
        for row in paths:
            road = []
            i = 0
            while i < len(row) - 1:
                road.append(self._mat[row[i]][row[i + 1]].id)
                i += 1
            # p = Path(vi, vj, road)
            # roads.append(p)
            roads.append(road)

        return roads

    @staticmethod
    def _out_edges(row, unconn):
        edegs = []
        for i in range(len(row)):
            if row[i] != unconn:
                edegs.append((i, row[i]))
        return edegs

    def __str__(self):
        return "[\n" + ",\n".join(map(str, self._mat)) + "\n]" + "\nUnconnected: " + str(self._unconn)


class Path:
    def __init__(self, origin, destination, path):
        self.origin = origin
        self.destination = destination
        self.path = path

    def get_ori(self):
        return self.origin

    def get_dest(self):
        return self.destination

    def get_path(self):
        return self.path


if __name__ == '__main__':
    road_network = \
        [[0, Edge(0, 0, 1), Edge(1, 0, 2), Edge(2, 0, 3)],
         [Edge(3, 1, 0), 0, 0, Edge(4, 1, 3)],
         [Edge(5, 2, 0), 0, 0, Edge(6, 2, 3)],
         [Edge(7, 3, 0), Edge(8, 3, 1), Edge(9, 3, 2), 0]]
    gra = Graph(road_network)


def parse_roadnet(roadnetFile):
    roadnet = json.load(open(roadnetFile))

    lane_phase_info_dict = {}
    road_info_dict = {}

    for intersection in roadnet["intersections"]:
        # if intersection['virtual']:
        #     continue
        lane_phase_info_dict[intersection['id']] = {"start_lane": [],
                                                    "end_lane": [],
                                                    "lane_mapping": {}}
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
            # roadLink_lane_pair[ri].append((sl, el))
            if road_link['type'] != 'turn_right':
                roadLink_lane_pair[rik].append([sl, el])

            if road_link['type'] != 'turn_right':
                roadLink_lane_pair[rik].append(road_link['direction'])
                rik += 1

        lane_phase_info_dict[intersection['id']]["start_lane"] = sorted(list(set(start_lane)))
        lane_phase_info_dict[intersection['id']]["end_lane"] = sorted(list(set(end_lane)))
        lane_phase_info_dict[intersection['id']]["lane_mapping"] = roadLink_lane_pair

    for road in roadnet["roads"]:
        road_info_dict[road["id"]] = {"end_intersection": road["endIntersection"],
                                      "link_roads": []}
        link_roads = lane_phase_info_dict[road["endIntersection"]]["end_lane"]
        road_info_dict[road["id"]]["link_roads"] = link_roads

    return lane_phase_info_dict, road_info_dict
