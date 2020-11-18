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

import networkx as nx
import pandas as pd
import numpy as np
import math
import argparse
import json
import time
import sys
import os
import importlib
from tabulate import tabulate

from gridappsd import GridAPPSD

global xfm_df, load_df, der_df, exit_flag
global logfp

def callback(headers, message):
    global exit_flag, xfm_df, load_df, der_df
    fr = message['FROM']
    to = message['TO']
    print('\nTRANSFORMER_CAPACITY microservice response received', flush=True)
    print('\nTRANSFORMER_CAPACITY microservice response received', file=logfp)
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
        xfm_name = xfm_dict['pname']
        index = xfm_df.pname[xfm_df.pname == xfm_name].index.tolist()
        # Check if the transformer is a regulator. Look for ratedU.
        # Same value of ratedU on both sides means it is a regulator.
        isXFMR = set(xfm_df['ratedU'][index].tolist())
        if xfm_name not in checked and len(isXFMR) > 1:
            node = xfm_df.bus[index[1]]
            checked.append(xfm_name)
            des = [node]
            children = descendant(fr, to, node, des)
            totalP = 0.
            totalQ = 0.
            count_Load = 0
            count_DER = 0
            KVA_DER = 0
            for n in children:
                index_load = load_df.bus[load_df.bus == n].index.tolist()
                for k in index_load:
                    count_Load += 1
                    totalP += float(load_df.iloc[k, 3])/1000.
                    totalQ += float(load_df.iloc[k, 4])/1000.
                try:
                    index_der = der_df.bus[der_df.bus == n].index.tolist()
                    for k in index_der:
                        count_DER += 1
                        KVA_DER += float(der_df.iloc[k, 2])/1000.
                except:
                    pass
            loading_1 = math.sqrt(totalP**2 + totalQ**2)/(float(xfm_dict['ratedS'])/1000.)
            loading_2 = KVA_DER/(float(xfm_dict['ratedS'])/1000.)
            message = dict(NAME = xfm_name,                           
                           kVA_LOAD = math.sqrt(totalP**2 + totalQ**2),
                           TOTAL_LOAD = count_Load,
                           kVA_DER = KVA_DER,
                           TOTAL_DER = count_DER,
                           XFMR_RATING = float(xfm_dict['ratedS'])/1000.,
                           LOADING = max(loading_1, loading_2))
            report_xfmr.append(message)

    xfmr_df = pd.DataFrame(report_xfmr)
    if xfmr_df.empty:
        print('TRANSFORMER_CAPACITY there are no Service Transformers in the selected feeder')
        print('TRANSFORMER_CAPACITY there are no Service Transformers in the selected feeder', file=logfp)
    else:
        print('TRANSFORMER_CAPACITY output:')
        print('TRANSFORMER_CAPACITY output:', file=logfp)
        print(tabulate(xfmr_df, headers = 'keys', tablefmt = 'psql'), flush=True)
        print(tabulate(xfmr_df, headers = 'keys', tablefmt = 'psql'), file=logfp)
        # Report based on loading. 
        loading_xfmr = []
        Loading = [x for x in xfmr_df['LOADING'] if x >= 0]
        normal = [l for l in Loading if l < 0.90]
        acceptable = [l for l in Loading if l >= 0.90 and l <=1]
        needatt = [l for l in Loading if l > 1]
        message = dict(VI = (len(Loading) - len(needatt))/len(Loading),                           
                    MINIMUM = min(Loading),
                    MAXIMUM = max(Loading),
                    AVERAGE = sum(Loading)/len(Loading))
        loading_xfmr.append(message)
        loading_df = pd.DataFrame(loading_xfmr)
        print('TRANSFORMER_CAPACITY report:')
        print('TRANSFORMER_CAPACITY report:', file=logfp)
        print(tabulate(loading_df, headers = 'keys', showindex = False, tablefmt = 'psql'), flush=True)
        print(tabulate(loading_df, headers = 'keys', showindex = False, tablefmt = 'psql'), file=logfp)
    exit_flag = True

def start(log_file, feeder_mrid, model_api_topic):
    global logfp
    logfp = log_file

    global xfm_df, load_df, der_df
    
    print("\nTRANSFORMER_CAPACITY starting!!!------------------------------------------------")
    print("\nTRANSFORMER_CAPACITY starting!!!------------------------------------------------", file=logfp)

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)

    # Get transformer data
    xfm_df = sparql_mgr.query_transformers()
    print('TRANSFORMER_CAPACITY transformer data obtained', flush=True)
    print('TRANSFORMER_CAPACITY transformer data obtained', file=logfp)
    
    load_df = sparql_mgr.query_energyconsumer()
    der_df = sparql_mgr.query_der()
    print('TRANSFORMER_CAPACITY load and DER data obtained', flush=True)
    print('TRANSFORMER_CAPACITY load and DER data obtained', file=logfp)

    # Subscribe to microservice for getting the graph information
    message = {"modelId": feeder_mrid,
                   "requestType": "GRAPH",
                   "modelType": "STATIC",
                   "resultFormat": "JSON"}
    out_topic = "/topic/goss.gridappsd.model-validator.graph.out"
    gapps.subscribe(out_topic, callback)

    in_topic = "/topic/goss.gridappsd.model-validator.graph.in"
    gapps.send(in_topic, message)
    print("TRANSFORMER_CAPACITY sent request to microservice; waiting for response\n", flush=True)
    print("TRANSFORMER_CAPACITY sent request to microservice; waiting for response\n", file=logfp)

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

    #global message_period
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", help="Simulation Request")
    #parser.add_argument("--message_period",
    #                    help="How often the sample app will send open/close capacitor message.",
    #                    default=DEFAULT_MESSAGE_PERIOD)

    opts = parser.parse_args()
    #message_period = int(opts.message_period)
    sim_request = json.loads(opts.request.replace("\'",""))
    feeder_mrid = sim_request["power_system_config"]["Line_name"]

    model_api_topic = "goss.gridappsd.process.request.data.powergridmodel"
    log_file = open('transformer_capacity.log', 'w')

    start(log_file, feeder_mrid, model_api_topic)

if __name__ == "__main__":
    _main()
