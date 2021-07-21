# ------------------------------------------------------------------------------
# Copyright (c) 2021, Battelle Memorial Institute All rights reserved.
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
# ------------------------------------------------------------------------------
"""
Created on July 15, 2021

@author: Gary Black, Shiva Poudel
"""""

import sys
import os
import argparse
import json
import importlib
import numpy as np
from tabulate import tabulate

from gridappsd import GridAPPSD


def greenCircle(colorFlag):
    return '\u001b[32m\u25cf\u001b[37m' if colorFlag else '\u25cb'


def redCircle(colorFlag):
    return '\u001b[31m\u25cf\u001b[37m' if colorFlag else '\u25cf'


def yellowCircle(colorFlag):
    return '\u001b[33m\u25cf\u001b[37m' if colorFlag else '\u25d1'


def start(log_file, feeder_mrid, model_api_topic):
    global logfile
    logfile = log_file

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)

    #ysparse,nodelist = sparql_mgr.ybus_export()

    #idx = 1
    #nodes = {}
    #for obj in nodelist:
    #    nodes[idx] = obj.strip('\"')
    #    idx += 1
    ##print(nodes)

    #Ybus = {}
    #for obj in ysparse:
    #    items = obj.split(',')
    #    if items[0] == 'Row':
    #        continue
    #    if nodes[int(items[0])] not in Ybus:
    #        Ybus[nodes[int(items[0])]] = {}
    #    Ybus[nodes[int(items[0])]][nodes[int(items[1])]] = complex(float(items[2]), float(items[3]))
    ##print(Ybus)

    Ysys = {}
    Unsupported = {}

    mod_import = importlib.import_module('line_model_validator.line_model_validator')
    start_func = getattr(mod_import, 'start')
    start_func(log_file, feeder_mrid, model_api_topic, False, Ysys, Unsupported)
    #print('line_model_validator Ysys...')
    #print(Ysys)
    #line_count = 0
    #for bus1 in Ysys:
    #    line_count += len(Ysys[bus1])
    #print('\nLine_model # entries: ' + str(line_count) + '\n', flush=True)
    #print('\nLine_model # entries: ' + str(line_count) + '\n', file=logfile)

    mod_import = importlib.import_module('power_transformer_validator.power_transformer_validator')
    start_func = getattr(mod_import, 'start')
    start_func(log_file, feeder_mrid, model_api_topic, False, Ysys, Unsupported)
    #print('power_transformer_validator Ysys...')
    #print(Ysys)
    #count = 0
    #for bus1 in Ysys:
    #    count += len(Ysys[bus1])
    #xfmr_count = count - line_count
    #print('Power_transformer # entries: ' + str(xfmr_count) + '\n', flush=True)
    #print('Power_transformer # entries: ' + str(xfmr_count) + '\n', file=logfile)

    mod_import = importlib.import_module('switching_equipment_validator.switching_equipment_validator')
    start_func = getattr(mod_import, 'start')
    start_func(log_file, feeder_mrid, model_api_topic, False, Ysys, Unsupported)
    #print('switching_equipment_validator (final) Ysys...')
    #print(Ysys)
    #count = 0
    #for bus1 in Ysys:
    #    count += len(Ysys[bus1])
    #switch_count = count - line_count - xfmr_count
    #print('Switching_equipment # entries: ' + str(switch_count) + '\n', flush=True)
    #print('Switching_equipment # entries: ' + str(switch_count) + '\n', file=logfile)

    #print('\n*** Full Ysys:\n')
    #for bus1 in Ysys:
    #    for bus2 in Ysys[bus1]:
    #        print(bus1 + ',' + bus2 + ',' + str(Ysys[bus1][bus2].real) + ',' + str(Ysys[bus1][bus2].imag))

    ysysCount = 0
    for bus1 in Ysys:
        ysysCount += len(Ysys[bus1])
    #print('Total computed # entries: ' + str(ysysCount) + '\n', flush=True)
    #print('Total computed # entries: ' + str(ysysCount) + '\n', file=logfile)

    # build the Numpy matrix from the full Ysys before we start deleting
    # entries to check Ysys vs. Ybus
    # first, create a node index dictionary
    loadNode = {}
    loadCount = 0
    for bus1 in list(Ysys):
        if bus1 not in loadNode:
            loadNode[bus1] = loadCount
            loadCount += 1
        for bus2 in list(Ysys[bus1]):
            if bus2 not in loadNode:
                loadNode[bus2] = loadCount
                loadCount += 1
    print(loadNode)
    print('loadNode size: ' + str(loadCount))

    loadYbus = np.zeros((loadCount,loadCount), dtype=complex)
    # next, remap into a numpy array
    for bus1 in list(Ysys):
        for bus2 in list(Ysys[bus1]):
            loadYbus[loadNode[bus2],loadNode[bus1]] = loadYbus[loadNode[bus1],loadNode[bus2]] = Ysys[bus1][bus2]

    np.set_printoptions(threshold=sys.maxsize)
    #print(loadYbus)

    sourcebus, vang = sparql_mgr.sourcebus_query()
    sourcebus = sourcebus.upper()
    print('query results sourcebus name: ' + sourcebus)
    print('query results vang: ' + vang)

    src_idxs = []
    if sourcebus+'.1' in loadNode:
        src_idxs.append(loadNode[sourcebus+'.1'])
    if sourcebus+'.2' in loadNode:
        src_idxs.append(loadNode[sourcebus+'.2'])
    if sourcebus+'.3' in loadNode:
        src_idxs.append(loadNode[sourcebus+'.3'])
    print('src_idxs: ' + str(src_idxs))

    bindings = sparql_mgr.nomv_query()
    print(bindings)



def _main():
    # for loading modules
    if (os.path.isdir('shared')):
        sys.path.append('.')
    elif (os.path.isdir('../shared')):
        sys.path.append('..')

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", help="Simulation Request")

    opts = parser.parse_args()
    sim_request = json.loads(opts.request.replace("\'",""))
    feeder_mrid = sim_request["power_system_config"]["Line_name"]

    model_api_topic = "goss.gridappsd.process.request.data.powergridmodel"
    log_file = open('ysystem_validator.log', 'w')

    start(log_file, feeder_mrid, model_api_topic)


if __name__ == "__main__":
    _main()

