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
Created on Sept 8, 2020

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

from gridappsd import GridAPPSD


def start(feeder_mrid, model_api_topic):
    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')
    GLMManager = getattr(importlib.import_module('shared.glm'), 'GLMManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid)

    # Get Service Transformer and EnergyConsumer data
    xfm_df = sparql_mgr.query_transformers()
    print('Service transformer data obtained')

    load_df = sparql_mgr.query_energyconsumer()
    print('Load data obtained')

    # Get the base glm file representing a network model
    model = sparql_mgr.get_glm()
    glm_mgr = GLMManager(model=model, model_is_path=False)
    model_dict = glm_mgr.model_dict
    print('Base network model obtained')

    # Form a graph G(V,E)
    G = nx.Graph()  
    nor_open = []
    for key, value in model_dict.items():
        if value['object'] == 'switch' and value['status'] =='OPEN':
            nor_open.append(value['name'])
    
    # Screen throuh model_dict of a feeder and form a graph using all possible connecting elements
    component = ['transformer', 'regulator', 'triplex_line', 'overhead_line', 'switch', 'recloser', 'fuse']
    for key, value in model_dict.items():
        if value['object'] in component and value['name'] not in nor_open:
            G.add_edge(value['from'].replace('"', ''), value['to'].replace('"', ''))
    print('The graph information--> Number of Nodes:', G.number_of_nodes(), 'and', " Number of Edges:", G.number_of_edges(), "\n")
    T = list(nx.bfs_tree(G, source = '650').edges())
    Nodes = list(nx.bfs_tree(G, source = '650').nodes())
    fr, to = zip(*T)
    fr = list(fr)
    to = list(to) 

    # A recursive function to find all descendent of a given node
    def descendant (fr, to, node, des):
        ch = [n for n, e in enumerate(fr) if e == node]
        for k in ch:
            des.append(to[k])
            node = to[k]
            descendant(fr, to, node, des)
        return des

    # Pick a node, find all descendent and sum all loads to check against the rating 
    checked = []
    report_xfmr = []
    for xfm in xfm_df.itertuples(index=False):
        xfm_dict = xfm._asdict()
        xfm_name = xfm_dict['name']
        if xfm_name not in checked and 'reg' not in xfm_name:
            index = xfm_df.name[xfm_df.name == xfm_name].index.tolist()
            node = xfm_df.bus[max(index)]
            checked.append(xfm_name)
            des = [node]
            children = descendant(fr, to, node, des)
            totalP = 0.
            totalQ = 0.
            count = 0
            for n in children:
                index = load_df.bus[load_df.bus == n].index.tolist()
                for k in index:
                    count += 1
                    totalP += float(load_df.iloc[k, 3])/1000.
                    totalQ += float(load_df.iloc[k, 4])/1000.
            message = dict(name = xfm_name,                           
                           kVA_total = math.sqrt(totalP**2 + totalQ**2),
                           tot_loads = count)
            report_xfmr.append(message)
    print(pd.DataFrame(report_xfmr))
    return


def _main():
    # for loading modules
    if (os.path.isdir('shared')):
        sys.path.append('.')
    elif (os.path.isdir('../shared')):
        sys.path.append('..')

    #_log.debug("Starting application")
    print("Application starting!!!-------------------------------------------------------")
    #global message_period
    parser = argparse.ArgumentParser()
    #parser.add_argument("simulation_id",
    #                    help="Simulation id to use for responses on the message bus.")
    parser.add_argument("--request", help="Simulation Request")
    #parser.add_argument("--message_period",
    #                    help="How often the sample app will send open/close capacitor message.",
    #                    default=DEFAULT_MESSAGE_PERIOD)

    opts = parser.parse_args()
    #listening_to_topic = simulation_output_topic(opts.simulation_id)
    #message_period = int(opts.message_period)
    sim_request = json.loads(opts.request.replace("\'",""))
    #model_mrid = sim_request["power_system_config"]["Line_name"]
    #_log.debug("Model mrid is: {}".format(model_mrid))
    #gapps = GridAPPSD(opts.simulation_id, address=utils.get_gridappsd_address(),
    #                  username=utils.get_gridappsd_user(), password=utils.get_gridappsd_pass())

    feeder_mrid = sim_request["power_system_config"]["Line_name"]
    model_api_topic = "goss.gridappsd.process.request.data.powergridmodel"

    start(feeder_mrid, model_api_topic)

if __name__ == "__main__":
    _main()
