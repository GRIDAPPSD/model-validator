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

from gridappsd import GridAPPSD, topics
from gridappsd.topics import simulation_output_topic, simulation_log_topic

global G, undirected_graph, loadbreaksw, exit_flag, measid_lbs, sw_status

def on_message(headers, message):
    global exit_flag
    print('Response: ' + str(message), flush= True)
    exit_flag = True
        
        
def get_topology(feeder_mrid, model_api_topic):

    global G, measid_lbs, loadbreaksw, undirected_graph  

    gapps = GridAPPSD()
    #TODO don't hardwire modelID to the 123-node model
    message = {"modelId": "_C1C3E687-6FFD-C753-582B-632A27E28507",
                   "requestType": "LOOPS",
                   "modelType": "STATIC",
                   "resultFormat": "JSON"}
    out_topic = "/topic/goss.gridappsd.model-validator.topology.out"
    gapps.subscribe(out_topic, on_message)

    in_topic = "/topic/goss.gridappsd.model-validator.topology.in"
    gapps.send(in_topic, message)
    print("Sent request to microservice; waiting for response\n", flush = True)
    
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
    print("\n\nTopology starting!!!------------------------------------------------------------")
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", help="Simulation Request")

    opts = parser.parse_args()
    #listening_to_topic = simulation_output_topic(opts.simulation_id)
    sim_request = json.loads(opts.request.replace("\'",""))
    feeder_mrid = sim_request["power_system_config"]["Line_name"]
    #_log.debug("Feeder mrid is: {}".format(feeder_mrid))

    model_api_topic = "goss.gridappsd.process.request.data.powergridmodel"
    get_topology(feeder_mrid, model_api_topic)

if __name__ == "__main__":
    _main()

