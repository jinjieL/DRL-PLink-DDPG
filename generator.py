# -*- coding: utf-8 -*-
'''
generate the traffic trace

#Trace format: <flow_id, time_interval, flow_size(KB), flow_type, src, dst>

'''

import os
import sys
import string
import time
import random
import math
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from time import sleep, time


# """ This function is taken from http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/ """
def weightedChoice(flowWeights):
    totals = []
    runningTotal = 0

    for w in flowWeights:
        runningTotal += w
        totals.append(runningTotal)

    rnd = random.random() * runningTotal
    for i, total in enumerate(totals):
        if rnd < total:
            return i


def randomSize(flowSizes, flowWeights, threshold_size):
    index = weightedChoice(flowWeights)
    flowsize = flowSizes[index]

    if flowsize <= threshold_size:
    #if flowsize <=  1000:
        flowtype = 1
    else:
        flowtype = random.randint(2,3)
    return [flowsize, flowtype]


# http://preshing.com/20111007/how-to-generate-random-timings-for-a-poisson-process/
# Return next time interval (ms)
def nextTime(rateParameter):
    # We use round function to get a integer
    a = -math.log(1.0 - random.random())
    b = rateParameter
    c = round(a / b)
    # return round(-math.log(1.0 - random.random()) / rateParameter)
    return -math.log(1.0 - random.random()) / rateParameter


def trace_generate(cdf_filename, load, capacity, flow_num, host_num):
    # print(cdf_filename,load,capacity,flow_num,host_num)
    flowCDF = []
    with open(cdf_filename, 'r') as f:
        for l in f.readlines():
            flowCDF.append(([float(i) for i in str.split(l)]))
    flowType = cdf_filename.split('.')[0]
    # print(flowType)
    # print(flowCDF)

    flowSizes= []
    flowWeights = []
    prev = 0
    for size in flowCDF:
        flowSizes.append(int(size[0]))
        flowWeights.append(size[2] - prev)
        prev = size[2]
    # print(flowSizes)
    # print(flowWeights)

    if cdf_filename == 'datamining.txt':
        threshold_size = 10
    elif cdf_filename == 'websearch.txt':
        threshold_size = 1000
    flows = []
    for i in range(flow_num):
        flows.append(randomSize(flowSizes, flowWeights, threshold_size))
    # print(flows)

    # Get average throughput
    throughput = load * capacity
    # Get average flow size
    total_size = 0
    for flow in flows:
        total_size += flow[0]
    avg = total_size / len(flows)

    # Get average number of requests per second
    num = throughput * 1024 * 1024 / (avg * 1024 * 8)
    # Get average request rate (number of requests every 1 ms)
    rate = num / 1000
    # print(rate)

    hostlist = range(1, host_num+1)

    # Generate time interval
    times = []
    for i in range(flow_num):
        # Get time interval (sleep time)
        times.append(nextTime(rate))

    print times
    # Trace format: <time_interval, flow_size(KB), flow_type, deadline(ms), src, dst>
    trace = []

    for i in range(flow_num):
        flow = []
        flow_size = flows[i][0]
        flow_type = flows[i][1]

        # choose src and dit randomly
        hostslice = random.sample(hostlist, 2)  # 从list中随机获取2个元素，作为一个list返回
        host_src = hostslice[0]
        host_dst = hostslice[1]
        dsport = random.randint(49152, 65535)

        deadline = 0
        # Type 1 flows
        if flow_type == 1:
            # Calculate ideal FCT (ms)
            #ideal_fct = 0.2 + float(flow_size) * 8 * 1024 / (capacity * 1024 * 1024) * 1000
            if threshold_size==10:
                ideal_fct = 1 + float(flow_size) * 1024 / (capacity * 1024 * 1024) * 1000
            elif threshold_size==1000:
                ideal_fct = 20 + float(flow_size) * 1024 / (capacity * 1024 * 1024) * 1000
            # Deadline is assigned to
            deadline = int(math.ceil(2 * ideal_fct))
        flow = [times[i], flow_size, flow_type, deadline, host_src, host_dst, dsport]
        trace.append(flow)
    return trace


# CDF_file = 'datamining.txt'
# load = 0.8
# capacity = 1000  # (Mbps)
# flownum = 40
# hostnum = 4
# tm = trace_generate(CDF_file, load, capacity, flownum, hostnum)
# print tm
# ideal_fct = 1 + float(1) * 1024 / (capacity * 1024 * 1024) * 1000
# deadline = int(math.ceil(2 * ideal_fct))
# print deadline