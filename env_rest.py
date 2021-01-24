# -*- coding: utf-8 -*-

import requests
import numpy as np
import json
import networkx as nx
import matplotlib.pyplot as plt
import time
import os


class NetworkEnvRestAPI(object):

    def __init__(self):

        self.link_to_port = {}  # {(src_dpid,dst_dpid):(src_port,dst_port),}
        self.access_table = {}  # {(sw,port):(ip, mac),} MAC地址表
        self.switch_port_table = {}  # {dpid:set(port_num,),}
        self.access_ports = {}  # {dpid:set(port_num,),}
        self.interior_ports = {}  # {dpid:set(port_num,),}
        self.switches = []  # self.switches = [dpid,]
        self.switches_name = {}  # self.switches_name = {dpid:name}
        self.shortest_paths = {}  # {dpid:{dpid:[[path],],},}
        self.graph = nx.DiGraph()  # 创建有向图

        # self.rest_api()
        # self.shortest_paths = self.all_k_shortest_paths(self.graph, weight='weight', k=2)

    def rest_api(self):
        ##########################################
        # 交换机id、端口信息
        ##########################################
        url1 = 'http://127.0.0.1:8080/v1.0/topology/switches'
        r1 = requests.get(url1)
        # print(r1.encoding)
        # r1.encoding = 'utf-8'
        switchs_list = r1.json()
        # print switchs_list

        for sw in switchs_list:
            id = int(str(sw['dpid']).lstrip("0"), 16)
            dpid = str(sw['dpid'])
            # print id
            self.switches.append(id)
            # print str(sw['ports'][0]['name']).split('-')[0]
            self.switches_name[id] = str(sw['ports'][0]['name']).split('-')[0]
            self.switch_port_table[dpid] = {}
            # self.switch_port_table.setdefault(dpid, {})
            # switch_port_table is equal to interior_ports plus access_ports.
            # interior_ports.setdefault(dpid, {})
            # access_ports.setdefault(dpid, {})
            # self.switch_port_table[dpid]['port'] = {}
            for port in sw['ports']:
                # print port
                port_no = str(port['port_no']).lstrip("0")
                # switch_port_table[dpid]['port'].setdefault(port_no, str(port['name']))
                # self.switch_port_table[dpid]['port'][port_no] = str(port['name'])
                self.switch_port_table[dpid][port_no] = str(port['name'])
            # self.switch_port_table[dpid]['dpid'] = str(sw['dpid'])

        # print self.switch_port_table
        self.switch_port_table = sorted(self.switch_port_table.items(), key=lambda x: x[0])
        print self.switch_port_table


        #######################################
        # 建立链路信息
        #######################################
        url = 'http://localhost:8080/v1.0/topology/links'
        r = requests.get(url)
        link_list = r.json()
        # print len(link_list)

        for link in link_list:
            # print link
            src_id = int(str(link['src']['dpid']).lstrip('0'), 16)
            # src_id = str(link['src']['name']).split('-')[0]
            src_port = int(str(link['src']['port_no']).lstrip('0'))
            dst_id = int(str(link['dst']['dpid']).lstrip('0'), 16)
            # dst_id = str(link['dst']['name']).split('-')[0]
            dst_port = int(str(link['dst']['port_no']).lstrip('0'))
            # print src_id,dst_id
            # print src_port,dst_port
            self.link_to_port[(src_id, dst_id)] = (src_port, dst_port)
        # print self.link_to_port
        # {(3, 2): (2, 2), (1, 3): (2, 1), (3, 1): (1, 2), (1, 4): (3, 1), (2, 3): (2, 2), (4, 2): (2, 3), (4, 1): (1, 3), (2, 4): (3, 2)}
        # {('s1', 's3'): (2, 1), ('s2', 's4'): (3, 2), ('s2', 's3'): (2, 2), ('s3', 's1'): (1, 2), ('s4', 's1'): (1, 3), ('s3', 's2'): (2, 2), ('s1', 's4'): (3, 1), ('s4', 's2'): (2, 3)}

    #######################################
    # 建立有向图
    #######################################
    def get_graph(self):
        """
        	Get Adjacency matrix from link_to_port.
        """
        link_list = self.link_to_port.keys()
        self._graph = self.graph.copy()
        for src in self.switches:
            for dst in self.switches:
                if src == dst:
                    self._graph.add_edge(src, dst, weight=0)
                elif (src, dst) in link_list:
                    self._graph.add_edge(src, dst, weight=1)
                else:
                    pass
        return self._graph

    #################################
    # 所有端点的k-path
    #################################
    def k_shortest_paths(self,graph, src, dst, weight='weight', k=5):
        """
            Creat K shortest paths from src to dst.
            generator produces lists of simple paths, in order from shortest to longest.
            创建从src到dst的K个最短路径。
            生成器按照从最短到最长的顺序生成简单路径列表。
        """
        generator = nx.shortest_simple_paths(graph, source=src, target=dst, weight=weight)
        shortest_paths = []
        try:
            for path in generator:
                if k <= 0:
                    break
                shortest_paths.append(path)
                k -= 1
            return shortest_paths
        except:
            print ("No path between %s and %s" % (src, dst))

    def all_k_shortest_paths(self, graph, weight='weight', k=5):
        """
        	Creat all K shortest paths between datapaths.
        	Note: We get shortest paths for bandwidth-sensitive
        	traffic from bandwidth-sensitive switches.
        	在数据路径之间创建所有K个最短路径。
        	注意：我们从带宽敏感交换机为带宽敏感流量获得最短路径。
        """
        _graph = graph.copy()
        paths = {}
        # Find k shortest paths in graph.
        for src in _graph.nodes():
            paths.setdefault(src, {src: [[src] for i in xrange(k)]})
            for dst in _graph.nodes():
                if src == dst:
                    continue
                paths[src].setdefault(dst, [])
                paths[src][dst] = self.k_shortest_paths(_graph, src, dst, weight=weight, k=k)
        return paths

    #######################################
    # 接入ovsdb
    #######################################
    def access_ovsdb(self):
        sw = self.switch_port_table
        os.system('sudo ovs-vsctl set-manager ptcp:6632')
        for isw in xrange(len(self.switch_port_table)):
            url = 'http://127.0.0.1:8080/v1.0/conf/switches/%s/ovsdb_addr' % (sw[isw][0])
            print url
            payload = json.dumps("tcp:127.0.0.1:6632")
            r = requests.put(url, data=payload)

    #######################################
    # all设置Qos rule
    #######################################
    def set_qos_rule(self):
        url = 'http://localhost:8080/qos/rules/all'
        for i in range(3):
            param = {
                        "match": {
                            "ip_dscp": i,
                        },
                        "actions":{
                            "queue": "%d" %(i)

                        }

                    }
            payload = json.dumps(param)
            r = requests.post(url, data=payload)

    #######################################
    # all设置Queue
    #######################################
    def set_queue_all(self, r0, r1, r2, BW):
        # BW = 1000000000     # (1G)
        # BW = 10000000000
        BW0 = BW * r0
        BW1 = BW * r1
        BW2 = BW * r2

        url = 'http://localhost:8080/qos/queue/all'
        param = {
            "type": "linux-htb",
            "max_rate": "%d" %(BW),
            "queues":
                [
                    {"max_rate": "%d" %(BW0)},
                    {"max_rate": "%d" %(BW1)},
                    {"max_rate": "%d" %(BW2)}
                ]
        }
        payload = json.dumps(param)
        r = requests.post(url, data=payload)

    def set_queue(self, rate, BW):
        # BW = 1000000000     # (1G)
        # BW = 10000000000
        # BW0 = BW * r0
        # BW1 = BW * r1
        # BW2 = BW * r2

        pointer = 0
        sw = self.switch_port_table
        for isw in xrange(len(self.switch_port_table)):
            url = 'http://localhost:8080/qos/queue/%s' % (sw[isw][0])
            print url
            port_name = sorted(sw[isw][1].values())
            print port_name
            for iport in xrange(len(sw[isw][1])):
                print port_name[iport]
                BW0 = BW * rate[pointer]
                BW1 = BW * rate[pointer+1]
                BW2 = BW * rate[pointer+2]
                param = {
                    #"port_name": "1001-eth1",
                    "port_name": "%s" % (port_name[iport]),
                    "type": "linux-htb",
                    "max_rate": "%d" % (BW),
                    "queues":
                        [
                            {"max_rate": "%d" % (BW0)},
                            {"max_rate": "%d" % (BW1)},
                            {"max_rate": "%d" % (BW2)}
                        ]
                }
                payload = json.dumps(param)
                r = requests.post(url, data=payload)
                pointer = pointer + 3

    def Add_flow_entry(self, dpid, ipv4_src, ipv4_dst, port, Flag):
        url = 'http://localhost:8080/stats/flowentry/add'
        dpid = dpid

        matcharp={
                  "arp_spa": ipv4_src,
                  "arp_tpa": ipv4_dst,
                  "eth_type": 2054
               }

        param1 = {
                    "dpid": dpid,
                    "cookie": 1,
                    "cookie_mask": 1,
                    "table_id": 0,
                    "idle_timeout": 60,
                    "hard_timeout": 60,
                    "priority": 1,
                    "flags": 1,
                    "match": matcharp,
                    "actions": [
                        {
                            "type": "OUTPUT",
                            "port": port
                        }
                    ]
                 }

        payload1 = json.dumps(param1)
        r = requests.post(url, data=payload1)

        matchip = {
                    "ipv4_src": ipv4_src,
                    "ipv4_dst": ipv4_dst,
                    "ip_dscp": Flag,
                    "eth_type": 2048,
                    }

        param = {
                    "dpid": dpid,
                    "cookie": 1,
                    "cookie_mask": 1,
                    "table_id": 1,
                    "idle_timeout": 60,
                    "hard_timeout": 60,
                    "priority": 1,
                    "flags": 1,
                    "match": matchip,
                    "actions": [
                        {
                            "type": "OUTPUT",
                            "port": port
                        }
                    ]
                 }

        payload = json.dumps(param)
        r = requests.post(url, data=payload)


# envrest= NetworkEnvRestAPI()
# envrest.rest_api()
# graph = envrest.get_graph()
# paths = envrest.all_k_shortest_paths(graph, weight='weight', k=2)
# print paths

# envrest.access_ovsdb()
# time.sleep(5)
# envrest.set_qos_rule()
# envrest.set_queue_all(0.157888523694, 0.570586344206, 0.2715251321)

# envrest.Add_flow_entry(dpid=1, ipv4_src="10.0.0.1", ipv4_dst="10.0.0.2", port=)


