# -*- coding: utf-8 -*-
import numpy as np
import copy
import collections as col

from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import Link, Intf, TCLink, OVSLink
from mininet.topo import Topo

from generator import trace_generate
from env_rest import NetworkEnvRestAPI
import logging
import time
import os
from subprocess import Popen
from multiprocessing import Process
import pandas as pd
import math
import random


class Fattree(Topo):
    """
        Class of Fattree Topology.
    """
    CoreSwitchList = []
    AggSwitchList = []
    EdgeSwitchList = []
    HostList = []

    def __init__(self, k, density):
        self.pod = k
        self.density = density
        self.iCoreLayerSwitch = (k/2)**2
        self.iAggLayerSwitch = k*k/2
        self.iEdgeLayerSwitch = k*k/2
        self.iHost = self.iEdgeLayerSwitch * density

        # Init Topo
        Topo.__init__(self)

    def createNodes(self):
        self.createCoreLayerSwitch(self.iCoreLayerSwitch)
        self.createAggLayerSwitch(self.iAggLayerSwitch)
        self.createEdgeLayerSwitch(self.iEdgeLayerSwitch)
        self.createHost(self.iHost)

    # Create Switch and Host
    def _addSwitch(self, number, level, switch_list):
        """
            Create switches.
        """
        for i in xrange(1, number+1):
            PREFIX = str(level) + "00"
            if i >= 10:
                PREFIX = str(level) + "0"
            switch_list.append(self.addSwitch(PREFIX + str(i)))

    def createCoreLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 1, self.CoreSwitchList)

    def createAggLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 2, self.AggSwitchList)

    def createEdgeLayerSwitch(self, NUMBER):
        self._addSwitch(NUMBER, 3, self.EdgeSwitchList)

    def createHost(self, NUMBER):
        """
            Create hosts.
        """
        for i in xrange(1, NUMBER+1):
            if i >= 100:
                PREFIX = "h"
            elif i >= 10:
                PREFIX = "h0"
            else:
                PREFIX = "h00"
            self.HostList.append(self.addHost(PREFIX + str(i), cpu=1.0/NUMBER))

    def createLinks(self):
        """
            Add network links.
        """
        # Core to Agg
        end = self.pod/2
        for x in xrange(0, self.iAggLayerSwitch, end):
            for i in xrange(0, end):
                for j in xrange(0, end):
                    self.addLink(
                        self.CoreSwitchList[i*end+j],
                        self.AggSwitchList[x+i])   # use_htb=False

        # Agg to Edge
        for x in xrange(0, self.iAggLayerSwitch, end):
            for i in xrange(0, end):
                for j in xrange(0, end):
                    self.addLink(
                        self.AggSwitchList[x+i], self.EdgeSwitchList[x+j])   # use_htb=False

        # Edge to Host
        for x in xrange(0, self.iEdgeLayerSwitch):
            for i in xrange(0, self.density):
                self.addLink(
                    self.EdgeSwitchList[x],
                    self.HostList[self.density * x + i])   # use_htb=False

    def set_ovs_protocol_13(self,):
        """
            Set the OpenFlow version for switches.
        """
        self._set_ovs_protocol_13(self.CoreSwitchList)
        self._set_ovs_protocol_13(self.AggSwitchList)
        self._set_ovs_protocol_13(self.EdgeSwitchList)

    def _set_ovs_protocol_13(self, sw_list):
        for sw in sw_list:
            cmd = "sudo ovs-vsctl set bridge %s protocols=OpenFlow13" % sw
            os.system(cmd)


def create_subnetList(topo, num):
    """
        Create the subnet list of the certain Pod.
    """
    subnetList = []
    remainder = num % (topo.pod/2)
    if topo.pod == 4:
        if remainder == 0:
            subnetList = [num-1, num]
        elif remainder == 1:
            subnetList = [num, num+1]
        else:
            pass
    elif topo.pod == 8:
        if remainder == 0:
            subnetList = [num-3, num-2, num-1, num]
        elif remainder == 1:
            subnetList = [num, num+1, num+2, num+3]
        elif remainder == 2:
            subnetList = [num-1, num, num+1, num+2]
        elif remainder == 3:
            subnetList = [num-2, num-1, num, num+1]
        else:
            pass
    else:
        pass
    return subnetList


def set_host_ip(net, topo):
    hostlist = []
    for k in xrange(len(topo.HostList)):
        hostlist.append(net.get(topo.HostList[k]))
    i = 1
    j = 1
    for host in hostlist:
        host.setIP("10.%d.0.%d" % (i, j))
        j += 1
        if j == topo.density+1:
            j = 1
            i += 1


def pingTest(net):
    """
        Start ping test.
    """
    net.pingAll()


def makeHostList(net):
    HostIPList = []
    HostList = []
    HostNameIPMAC = {}
    # print net.keys()
    for hostStr in net.keys():
        if "h" in hostStr:
            host = net.get(hostStr)
            HostList.append(host)
            HostIPList.append(host.IP())
            HostNameIPMAC[hostStr] = (host.IP(), host.MAC())
    return HostList, HostIPList, HostNameIPMAC


def makeAccess_table(net, HostNameIPMAC):
    Access_table = {}
    nodes = net.values()
    for node in nodes:
        if "h" in node.name:
            # print node.name
            # print node.intfList()
            for intf in node.intfList():
                # print (' %s:' % intf)
                if intf.link:
                    # print intf.link
                    intfs = [intf.link.intf1, intf.link.intf2]
                    intfs.remove(intf)
                    dpid_port = str(intfs[0]).split('-')
                    Access_table[(dpid_port[0], dpid_port[1].replace('eth', ''))] = HostNameIPMAC[node.name]
    return Access_table


def install_proactive(Access_table):
    # print Access_table
    for k, v in Access_table.items():
        #print k,v
        sw = k[0]
        port = k[1]
        ip = v[0]
        # print sw,port,ip
        cmd = "ovs-ofctl add-flow %s -O OpenFlow13 'table=0,idle_timeout=0,hard_timeout=0,priority=1,arp,nw_dst=%s,actions=output:%s'"\
              %(sw,ip,port)
        #print cmd
        os.system(cmd)
        cmd = "ovs-ofctl add-flow %s -O OpenFlow13 'table=1,idle_timeout=0,hard_timeout=0,priority=1,ip,nw_dst=%s,actions=output:%s'"\
              %(sw,ip,port)
        #print cmd
        os.system(cmd)


def run_ryu_rest():
    print 'ok'
    time.sleep(1)
    proc = Popen("ryu-manager "
                 "./utils/rest_conf_switch.py "
                 "./utils/rest_topology.py "
                 "./utils/rest_qos.py "
                 "./utils/ofctl_rest.py --observe-links", shell=True)
    # proc.wait()
    time.sleep(1)


def IPtoInteger(a):
    Integer = lambda x: sum([256 ** j * int(i) for j, i in enumerate(x.split('.')[::-1])])
    return Integer(a)


def rl_to_matrix(path, nodes):
    M = - np.ones((nodes, nodes))
    x = 0
    for i in range(nodes - 1):
        for j in range(i+1, nodes):
            M[i][j] = path[x]
            x = x + 1
    return M


def rl_state(env):
    s = []
    for t in env.trace:
        s.append(t[1:-1])
    s = np.array(s).flatten()
    return s


class NetworkEnv(object):

    def __init__(self):

        self.CONTROLLER_IP = "127.0.0.1"
        self.CONTROLLER_PORT = 6653

        self.envrest = NetworkEnvRestAPI()

        self.pod = 4
        self.density = 1

        self.iCoreLayerSwitch = (self.pod / 2) ** 2             # 4
        self.iAggLayerSwitch = self.pod * self.pod / 2          # 8
        self.iEdgeLayerSwitch = self.pod * self.pod / 2         # 8
        self.iHost = self.iEdgeLayerSwitch * self.density       #
        self.iSwitch = self.iCoreLayerSwitch + self.iAggLayerSwitch + self.iEdgeLayerSwitch     # 20

        #self.CDF_file = 'datamining.txt'
        self.CDF_file = 'websearch.txt'
        self.load = 0.8
        self.capacity = 10000    # (Mbps)
        self.flownum = 40
        self.hostnum = 8
        self.kPath = 4
        self.BW = self.capacity * 1000000
        # self.BW = 1000000000     # (1G)

        self.hostnum = self.iHost

        self.nodes = self.hostnum                               # 8
        self.state_dim = self.flownum * 5                       #
        self.rate_dim = 3 * (self.iSwitch * self.pod - self.iHost)      # 3*(80-8)=3*72
        self.path_dim = 3 * (self.nodes**2-self.nodes)/2                # 3*28
        self.action_dim = self.rate_dim + self.path_dim                 #

        self.HostList = []
        self.HostIPList = []
        self.HostNameIPMAC = {}
        self.Access_table = {}
        self.trace = []

        self.FCTi = 0    # Flow Completion Time
        self.DMRi = 0   # Deadline Meet Rate
        self.Tputi = 1   # FCT / Size
        self.Sizei = 0

    def reset(self,workload):
        os.system('sudo mn -c')
        os.system('sudo ovs-vsctl --all destroy qos')
        os.system('sudo ovs-vsctl --all destroy queue')

        pod = 4
        density = 1
        # Create Topo.
        topo = Fattree(pod, density)
        topo.createNodes()
        topo.createLinks()
        # Start Mininet.
        CONTROLLER_IP = "127.0.0.1"
        CONTROLLER_PORT = 6653
        self.net = Mininet(topo=topo, link=Link, controller=None, autoSetMacs=True)
        self.net.addController(
            'controller', controller=RemoteController,
            ip=CONTROLLER_IP, port=CONTROLLER_PORT)
        self.net.start()

        # Set OVS's protocol as OF13.
        # topo.set_ovs_protocol_13()
        # Set hosts IP addresses.
        set_host_ip(self.net, topo)

        self.HostList, self.HostIPList, self.HostNameIPMAC = makeHostList(self.net)
        # print self.HostList, self.HostIPList, self.HostNameIPMAC
        self.Access_table = makeAccess_table(self.net, self.HostNameIPMAC)
        # print self.Access_table

        install_proactive(self.Access_table)
        # CLI(self.net)
        # self.net.stop()

        # Ryu Controller
        p = Process(target=run_ryu_rest())
        p.start()
        p.join()
        time.sleep(2)
        self.envrest.rest_api()
        self.envrest.access_ovsdb()
        time.sleep(2)
        self.envrest.set_qos_rule()
        self.envrest.get_graph()

        # traffic
        self.set_CDF_file(workload)
        self.generate_trace()

        state = rl_state(self)
        # print state
        # CLI(self.net)
        # self.net.stop()
        return state

    def get_host_location(self, host_ip):
        """
            Get host location info ((datapath, port)) according to the host ip.
            self.access_table = {(sw,port):(ip, mac),}
        """

        for key in self.Access_table.keys():
            if self.Access_table[key][0] == host_ip:
                return key

        return None

    def get_sw(self, src, dst):
        """
            Get pair of source and destination switches.
        """

        src_location = self.get_host_location(src)  # src_location = (dpid, port)
        src_sw = src_location[0]

        dst_location = self.get_host_location(dst)  # dst_location = (dpid, port)
        dst_sw = dst_location[0]

        if src_sw and dst_sw:
            return src_sw, dst_sw
        else:
            return None

    def get_path(self, src, dst, weight, k):

        # shortest_paths = self.envrest.shortest_paths
        graph = self.envrest.get_graph()
        shortest_paths = self.envrest.all_k_shortest_paths(graph, weight='weight', k=k)

        src = list(self.envrest.switches_name.keys())[list(self.envrest.switches_name.values()).index(src)]
        dst = list(self.envrest.switches_name.keys())[list(self.envrest.switches_name.values()).index(dst)]

        paths = shortest_paths[src][dst][weight]
        return paths

    def get_port_pair_from_link(self, link_to_port, src_dpid, dst_dpid):
        """
        	Get port pair of link, so that controller can install flow entry.
        	link_to_port = {(src_dpid,dst_dpid):(src_port,dst_port),}
        """
        if (src_dpid, dst_dpid) in link_to_port:
            return link_to_port[(src_dpid, dst_dpid)]
        else:
            print ("Link from dpid:%s to dpid:%s is not in links" %
                             (src_dpid, dst_dpid))
            return None

    def install_flow(self, link_to_port, path, flow_info):

        src_ip = flow_info[0]
        dst_ip = flow_info[1]
        Flag = flow_info[2]

        for i in range(len(path)-1):
            # src-->dst
            port = self.get_port_pair_from_link(link_to_port, path[i], path[i+1])[0]
            self.envrest.Add_flow_entry(path[i], src_ip, dst_ip, port, Flag)

            # dst-->src
            port = self.get_port_pair_from_link(link_to_port, path[i+1], path[i])[0]
            self.envrest.Add_flow_entry(path[i+1], dst_ip, src_ip, port, Flag)

    def step(self, action):
        # setting the max-rate of queue
        action_rate = action[0:self.rate_dim]
        print 'action_rate', action_rate
        self.envrest.set_queue(action_rate, self.BW)

        adim = self.path_dim/3
        # pMF = rl_to_matrix(np.clip(np.floor(action[3:adim + 3]), 0, 1), 2)
        pMF = rl_to_matrix(np.clip(np.floor(action[self.rate_dim:adim + self.rate_dim]), 0, self.kPath-1), self.nodes)
        pEP1 = rl_to_matrix(np.clip(np.floor(action[adim + self.rate_dim:adim*2 + self.rate_dim]), 0, self.kPath-1), self.nodes)
        pEP2 = rl_to_matrix(np.clip(np.floor(action[adim*2 + self.rate_dim:adim*3 + self.rate_dim]), 0, self.kPath-1), self.nodes)
        # print pMF
        # print pEP1
        # print pEP2

        # setting the path
        for t in range(3):
            if t == 0:
                path_matrix = pMF
                Flag = t
            elif t == 1:
                path_matrix = pEP1
                Flag = t
            elif t == 2:
                path_matrix = pEP2
                Flag = t
            # print Flag
            # print path_matrix

            for i, vector in enumerate(path_matrix):
                for j, value in enumerate(vector):
                    if value == -1:
                        continue
                    else:
                        ip_src = self.HostIPList[i]
                        ip_dst = self.HostIPList[j]
                        result = self.get_sw(ip_src, ip_dst)  # result = (src_sw, dst_sw)
                        src_sw, dst_sw = result[0], result[1]
                        path = self.get_path(src_sw, dst_sw, int(value), self.kPath)
                        print 'type', Flag, ip_src, '<-->', ip_dst, path
                        flow_info = (ip_src, ip_dst, Flag)
                        self.install_flow(self.envrest.link_to_port, path, flow_info)
        # CLI(self.net)
        # self.net.stop()

        self.test_trace()
        return self.get_state_reward()

    def get_state_reward(self):
        FCTi = self.FCTi
        DMRi = self.DMRi
        Sizei = self.Sizei
        Tputi = self.Tputi

        headers = ['src', 'dst', 'flowtype', 'flowsize', 'FCT', 'meetrate']

        flownum = self.flownum
        # data = pd.read_csv('1.csv', header=None)
        num = 0
        time.sleep(5)
        t1 = time.time()
        while True:
            try:
                data = pd.read_csv('./data/episode_data.csv', header=None, engine='python')
            except:
                continue
            if data.shape[0] == flownum:
                num = flownum
                break
            if time.time() - t1 >= 1:
                num = data.shape[0]
                add = ['0.0.0.0', '0.0.0.0', 0, 0, 0, None]
                for i in range(flownum - data.shape[0]):
                    df = pd.DataFrame([add])
                    df.to_csv("./data/episode_data.csv", mode='a', header=False, index=False)
                print '补充数据'
                break

        data = pd.read_csv('./data/episode_data.csv.csv', header=None)
        data.to_csv('./data/flow.csv', mode='a', index=False, header=None)
        data.columns = headers
        # print data.groupby(['flowtype']).size()[1]
        MFnum = len(data[data.flowtype == 1])       # flow number of MF
        EF1num = len(data[data.flowtype == 2])      # flow number of EF1
        EF2num = len(data[data.flowtype == 3])      # flow number of EF2

        EF_FCT = data[data['flowtype'] != 1]['FCT'].sum()
        EF_num = len(data[data['flowtype'] != 1])
        EF_avgFCT = data[data['flowtype'] != 1]['FCT'].sum() / len(data[data['flowtype'] != 1])     # the avgFCT of EF

        MFmeetratenum = len(data[data.meetrate==True])
        # data["Tput"] = data["flowsize"] / (data["FCT"] * 1000)
        data['src'] = data['src'].apply(IPtoInteger)
        data['dst'] = data['dst'].apply(IPtoInteger)

        # total flow sizes
        Sizei_ = data['flowsize'].sum()

        print '前一个状态:'
        print '总大小:', Sizei, '总FCT:', FCTi, '时限满足率:',DMRi
        #print 'Tput', Tputi
        FCTi_ = data.FCT.sum()
        DMRi_ = float(MFmeetratenum)/MFnum
        Tputi_ = FCTi_ / Sizei_
        #print '流数量:', num
        print '总大小:', Sizei_, '总FCT:', FCTi_, '时限满足率:', DMRi_
        print 'Tput_', Tputi_
        #print '1/Tput_', 1/Tputi_

        perform = [Sizei_, FCTi_, Tputi_, EF_FCT, EF_num, EF_avgFCT, DMRi_]

        self.FCTi = FCTi_
        self.DMRi = DMRi_
        self.Sizei = Sizei_
        self.Tputi = Tputi_

        # # reward
        # DRM = DMRi_ + DMRi
        # print DMRi_,DMRi
        # Tput = Tputi_ / Tputi
        # print 'Tput=Tputi_ / Tputi:', Tput

        # reward = - 0.5 * DRM * FCT * 1000  # r = - 1/2 * (DMRi_ + DMRi) * (avg_FCTi_ - avg_FCTi)
        # reward = - DRM * Tput * 1000000  # r = 1/2 * (DMRi_ + DMRi) * ( FCT_/Size_ - FCT/Size) * N
        # reward = 0
        # if DMRi_ > DMRi and Tputi_ < Tputi:
        #     reward = 10
        # elif DMRi_ > DMRi and Tputi_ > Tputi:
        #     reward = -1
        # elif DMRi_ < DMRi and Tputi_ < Tputi:
        #     reward = -1
        # elif DMRi_ < DMRi and Tputi_ > Tputi:
        #     reward = -10
        if DMRi_ >= 0.9:
            reward = -Tputi_ * 1000
        elif DMRi_ >= 0.8:
            reward = -Tputi_ * 10 * 1000
        elif DMRi_ >= 0.7:
            reward = -Tputi_ * 20 * 1000
        elif DMRi_ >= 0.6:
            reward = -Tputi_ * 30 * 1000
        elif DMRi_ < 0.6:
            reward = -Tputi_ * 40 * 1000

        self.generate_trace()
        new_state = rl_state(self)
        return new_state, reward, perform

    def generate_trace(self):
        self.trace = trace_generate(self.CDF_file, self.load, self.capacity,
                                    self.flownum, self.hostnum)
        print self.trace

    def test_trace(self):
        cmd = 'sudo python ./utils/delete_episode_data.py'
        os.system(cmd)

        for i, flow in enumerate(self.trace):
            time_interval = flow[0]  # (ms)
            time.sleep(time_interval / 1000)
            # time_interval_all += time_interval
            # print(time_interval)
            size = flow[1]
            # size_all += flow[1]

            type = flow[2]
            deadline = flow[3]
            src = flow[4] - 1
            dst = flow[5] - 1
            Hostsrc = self.HostList[flow[4] - 1]
            Hostdst = self.HostList[flow[5] - 1]
            # print Hostsrc.IP(), self.HostIPList[src], Hostdst.IP(), self.HostIPList[dst]
            dsport = flow[6]

            Hostdst.popen("python ./utils/server.py {} {}".format(dsport, type), shell=False)
            Hostsrc.popen("python ./utils/client.py {} {} {} {} {} {} {} {}"
                          .format(Hostdst.IP(), dsport, size, type, deadline, src, dst, Hostsrc.IP()), shell=False)
        # CLI(self.net)

    def set_CDF_file(self, workload):
        self.CDF_file = workload

