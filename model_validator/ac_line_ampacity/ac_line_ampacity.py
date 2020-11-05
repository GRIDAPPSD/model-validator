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
import numpy as np
import time
from tabulate import tabulate

from gridappsd import GridAPPSD
from gridappsd.topics import simulation_output_topic, simulation_log_topic

global df_acline_measA 
global exit_flag
global logfile


def on_message(headers, message):
    global df_acline_measA, exit_flag

    if type(message) == str:
            message = json.loads(message)

    if 'message' not in message:
        if message['processStatus']=='COMPLETE' or \
           message['processStatus']=='CLOSED':
            print('AC_LINE_AMPACITY End of Simulation', flush=True)
            print('AC_LINE_AMPACITY End of Simulation', file=logfile)
            exit_flag = True

    else:
        meas_value = message["message"]["measurements"]
        print('AC_LINE_AMPACITY checking ACLine Rating', flush=True)
        print('AC_LINE_AMPACITY checking ACLine Rating', file=logfile)
        try:
            for k in range (df_acline_measA.shape[0]):
                measid = df_acline_measA['measid'][k]
                pamp = meas_value[measid]['magnitude']
                df_acline_measA.loc[df_acline_measA.index == k, 'flow'] = pamp
            print('AC_LINE_AMPACITY output:', flush = True)
            print('AC_LINE_AMPACITY output:', file=logfile)
            print(tabulate(df_acline_measA, headers = 'keys', tablefmt = 'psql'), flush=True)
            print(tabulate(df_acline_measA, headers = 'keys', tablefmt = 'psql'), file=logfile)
        except:
            print('AC_LINE_AMPACITY simulation Output and Object MeasID Mismatch', flush=True)
            print('AC_LINE_AMPACITY simulation Output and Object MeasID Mismatch', file=logfile)
            exit_flag = True


def start(log_file, feeder_mrid, model_api_topic, simulation_id):
    global logfile
    logfile = log_file

    print("\nAC_LINE_AMPACITY starting!!!----------------------------------------------------")
    print("\nAC_LINE_AMPACITY starting!!!----------------------------------------------------", file=logfile)

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')
    GLMManager = getattr(importlib.import_module('shared.glm'), 'GLMManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)

    # AC Line segement rating check
    global df_acline_measA
    df_acline_measA = sparql_mgr.acline_measurements(logfile)    
    # Combine measurement mrids for 'A' and rating together
    df_acline_rating = sparql_mgr.acline_rating_query() 
    if df_acline_measA is not None:
        print('AC_LINE_AMPACITY ACLineSegment measurements obtained', flush=True)
        print('AC_LINE_AMPACITY ACLineSegment measurements obtained', file=logfile)
        df_acline_measA = df_acline_measA.assign(flow = np.zeros(df_acline_measA.shape[0]))   
        for r in df_acline_rating.itertuples(index=False):
            index = df_acline_measA.index[df_acline_measA['eqname'] == r.eqname].tolist()
            rating = r.val
            for k in index:
                df_acline_measA.loc[df_acline_measA.index == k, 'rating'] = rating
        print('AC_LINE_AMPACITY ACLineSegment rating obtained', flush=True)
        print('AC_LINE_AMPACITY ACLineSegment rating obtained', file=logfile)
        print('AC_LINE_AMPACITY df_acline_measA: ' + str(df_acline_measA), flush=True)
        print('AC_LINE_AMPACITY df_acline_measA: ' + str(df_acline_measA), file=logfile)
    else:
        return

    sim_output_topic = simulation_output_topic(simulation_id)
    sim_log_topic = simulation_log_topic(simulation_id)
    print('AC_LINE_AMPACITY simulation output topic from function: ' + sim_output_topic, flush=True)
    print('AC_LINE_AMPACITY simulation output topic from function: ' + sim_output_topic, file=logfile)
    print('AC_LINE_AMPACITY simulation log topic from function: ' + sim_log_topic, flush=True)
    print('AC_LINE_AMPACITY simulation log topic from function: ' + sim_log_topic, file=logfile)

    gapps.subscribe(topic = sim_output_topic, callback = on_message)
    gapps.subscribe(topic = sim_log_topic, callback = on_message)
    print('AC_LINE_AMPACITY subscribed to both output and log topics, waiting for messages', flush=True)
    print('AC_LINE_AMPACITY subscribed to both output and log topics, waiting for messages', file=logfile)

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", help="Simulation Request")
    parser.add_argument("--simid", help="Simulation ID")

    opts = parser.parse_args()
    sim_request = json.loads(opts.request.replace("\'",""))
    feeder_mrid = sim_request["power_system_config"]["Line_name"]
    simulation_id = opts.simid

    model_api_topic = "goss.gridappsd.process.request.data.powergridmodel"
    log_file = open('ac_line_ampacity.log', 'w')

    start(log_file, feeder_mrid, model_api_topic, simulation_id)    


if __name__ == "__main__":
    _main()
