# -------------------------------------------------------------------------------
# Copyright (c) 2017, Battelle Memorial Institute All rights reserved.
# Battelle Memorial Institute (hereinafter Battelle) hereby grants permission to any person or entity
# lawfully obtaining a copy of this software and associated documentation files (hereinafter the
# Software) to redistribute and use the Software in source and binary forms, with or without modification.
# Such person or entity may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and may permit others to do so, subject to the following conditions:
# Redistributions of source code must retain the above copyright notice, this list of conditions and the
# following disclaimers.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided with the distribution.
# Other than as used herein, neither the name Battelle Memorial Institute or Battelle may be used in any
# form whatsoever without the express written consent of Battelle.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
# General disclaimer for use with OSS licenses
#
# This material was prepared as an account of work sponsored by an agency of the United States Government.
# Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any
# of their employees, nor any jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for
# the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process
# disclosed, or represents that its use would not infringe privately owned rights.
#
# Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer,
# or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United
# States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed
# herein do not necessarily state or reflect those of the United States Government or any agency thereof.
#
# PACIFIC NORTHWEST NATIONAL LABORATORY operated by BATTELLE for the
# UNITED STATES DEPARTMENT OF ENERGY under Contract DE-AC05-76RL01830
# -------------------------------------------------------------------------------
"""
Created on Sept 22, 2020

@author: Shiva Poudel
"""""

#from shared.sparql import SPARQLManager
#from shared.glm import GLMManager

import networkx as nx
import pandas as pd
import math
import argparse
import json
import sys
import os
import importlib
import numpy as np
import time
from tabulate import tabulate

from gridappsd import GridAPPSD, topics, utils
from gridappsd.topics import simulation_output_topic, simulation_log_topic, service_output_topic

global undirected_graph, loadbreaksw, exit_flag, measid_lbs, sw_status, openSW, lock_flag, feeder_id

def find_all_cycles():
    global undirected_graph, G, openSW, feeder_id
    G = nx.Graph()     
    for g in undirected_graph:
        if g['eqname'] not in openSW:
            G.add_edge(g['bus1'], g['bus2'])    
    cycle_stack = []
    output_cycles = set()
    nodes=[list(i)[0] for i in nx.connected_components(G)]

    def get_hashable_cycle(cycle):
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi-1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
        return tuple(result)
    
    for start in nodes:
        if start in cycle_stack:
            continue
        cycle_stack.append(start)
        
        stack = [(start,iter(G[start]))]
        while stack:
            parent,children = stack[-1]
            try:
                child = next(children)
                
                if child not in cycle_stack:
                    cycle_stack.append(child)
                    stack.append((child,iter(G[child])))
                else:
                    i = cycle_stack.index(child)
                    if i < len(cycle_stack) - 2: 
                        output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                
            except StopIteration:
                stack.pop()
                cycle_stack.pop()

    output_cycles = list(output_cycles)  
    list_of_cycles = []
    ind = 0
    for k in output_cycles:
        switches = []  
        for i in range(len(k)):  
            j = (i + 1) % len(k)
            edge = {k[i], k[j]}
            for l in loadbreaksw:
                check = set([l['bus1'], l['bus2']])
                if check == edge:
                    switches.append(l['name'])
        loop = dict(index = ind,
                    nEdges = len(k),
                    nSwitches = len(switches),
                    switches = switches)
        list_of_cycles.append(loop)
        ind += 1
    cycles = {'feeder_id': feeder_id, 'total_loops': len(output_cycles), 'loops':  list_of_cycles}
    return cycles

def connected():
    global undirected_graph, G, openSW
    G = nx.Graph()     
    for g in undirected_graph:
        if g['eqname'] not in openSW:
            G.add_edge(g['bus1'], g['bus2'])    
    return nx.is_connected(G)

def on_message(headers, message):
    global lock_flag, openSW
    lock_flag = True
    Loadbreak = []
    meas_value = message['message']['measurements']
    for d in measid_lbs:                
        v = d['measid']
        p = meas_value[v]
        if p['value'] == 0:
            Loadbreak.append(d['eqname'])
    openSW = list(set(Loadbreak))
    print("The open switches are: ", openSW, flush= True)
    lock_flag = False
    
def handle_request(headers, message):

    global measid_lbs, loadbreaksw, undirected_graph, openSW, G
    while lock_flag:
        time.sleep(0.1)

    a = 'I got the request'
    gapps = GridAPPSD()
    print(a, flush= True)
    out_topic = "/topic/goss.gridappsd.model-validator.topology.out"
    print("I am sending the response", flush = True)
    if message['requestType'] == 'LOOPS':
        if message['modelType'] == 'STATIC':
            openSW = []
            response = find_all_cycles()
        else:
            response = find_all_cycles()
    elif message['requestType'] == 'ISOLATED_SECTIONS':
        response = {'Is the graph connected': connected()}
    else:
        response = 'Invalid request type'
    gapps.send(out_topic, response)

    

def check_topology(feeder_mrid, model_api_topic, simulation_id):

    global lock_flag, feeder_id
    lock_flag = False

    global measid_lbs, loadbreaksw, undirected_graph, openSW
    feeder_id = feeder_mrid
    openSW = []

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)
    
    # Get graph connectivity    
    undirected_graph = sparql_mgr.graph_query()
    sourcebus = sparql_mgr.sourcebus_query()
    print('Conectivity information obtained', flush = True)

    loadbreaksw = sparql_mgr.switch_query()
    measid_lbs = sparql_mgr.switch_meas_query()
    find_all_cycles()
    sim_output_topic = simulation_output_topic(simulation_id)
    in_topic = "/topic/goss.gridappsd.model-validator.topology.in"
    gapps.subscribe(in_topic, handle_request)
    gapps.subscribe(sim_output_topic, on_message)

    global exit_flag
    exit_flag = False

    while not exit_flag:
        time.sleep(0.1)
    

def _main():
    # for loading modules
    if (os.path.isdir('shared')):
        sys.path.append('.')
    elif (os.path.isdir('../shared')):
        sys.path.append('..')

    #_log.debug("Starting application")
    print("\n\nMicroservices starting!!!-------------------------------------------------------")
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", help="Simulation Request")
    parser.add_argument("--simid", help="Simulation ID")

    opts = parser.parse_args()
    #listening_to_topic = simulation_output_topic(opts.simulation_id)
    sim_request = json.loads(opts.request.replace("\'",""))
    feeder_mrid = sim_request["power_system_config"]["Line_name"]
    #_log.debug("Feeder mrid is: {}".format(feeder_mrid))
    simulation_id = opts.simid
    #_log.debug("Simulation ID is: {}".format(simulation_mrid))

    model_api_topic = "goss.gridappsd.process.request.data.powergridmodel"
    check_topology(feeder_mrid, model_api_topic, simulation_id)

if __name__ == "__main__":
    _main()

