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
Created on Dec 16, 2020

@author: Gary Black, Shiva Poudel
"""""

#from shared.sparql import SPARQLManager

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

global logfile


def start(log_file, feeder_mrid, model_api_topic):
    global logfile
    logfile = log_file

    print("\nLINE_MODEL_VALIDATOR starting!!!----------------------------------------------------")
    print("\nLINE_MODEL_VALIDATOR starting!!!----------------------------------------------------", file=logfile)

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)

    ysparse,nodelist = sparql_mgr.ybus_export()

    ybus = {}
    for obj in ysparse:
        items = obj.split(',')
        if items[0] == 'Row':
            continue
        if items[0] not in ybus:
            ybus[items[0]] = {}
        ybus[items[0]][items[1]] = complex(float(items[2]), float(items[3]))
    print(ybus)

    idx = 1
    nodes = {}
    for obj in nodelist:
        nodes[obj] = idx
        idx += 1
    print(nodes)

    bindings = sparql_mgr.perLengthImpedence_line_configs()
    #print('LINE_MODEL_VALIDATOR line_configs query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR line_configs query results:', file=logfile)
    #print(bindings, file=logfile)

    Zabc = {}
    for obj in bindings:
        line_config = obj['line_config']['value']
        count = int(obj['count']['value'])
        row = int(obj['row']['value'])
        col = int(obj['col']['value'])
        r_ohm_per_m = float(obj['r_ohm_per_m']['value'])
        x_ohm_per_m = float(obj['x_ohm_per_m']['value'])
        b_S_per_m = float(obj['b_S_per_m']['value'])
        #print('line_config: ' + line_config + ', count: ' + str(count) + ', row: ' + str(row) + ', col: ' + str(col) + ', r_ohm_per_m: ' + str(r_ohm_per_m) + ', x_ohm_per_m: ' + str(x_ohm_per_m) + ', b_S_per_m: ' + str(b_S_per_m))

        if line_config not in Zabc:
            if count == 1:
                Zabc[line_config] = np.zeros((1,1), dtype=complex)
            elif count == 2:
                Zabc[line_config] = np.zeros((2,2), dtype=complex)
            elif count == 3:
                Zabc[line_config] = np.zeros((3,3), dtype=complex)

        Zabc[line_config][row-1,col-1] = complex(r_ohm_per_m, x_ohm_per_m)
        if row != col:
            Zabc[line_config][col-1,row-1] = complex(r_ohm_per_m, x_ohm_per_m)

    for line_config in Zabc:
        print('Zabc[' + line_config + ']: ' + str(Zabc[line_config]))
    print('')

    bindings = sparql_mgr.perLengthImpedence_line_names()
    #print('LINE_MODEL_VALIDATOR line_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR line_names query results:', file=logfile)
    #print(bindings, file=logfile)

    perLengthImpedenceLinesPos = {}
    perLengthImpedenceLinesNeg = {}
    for obj in bindings:
        line_name = obj['line_name']['value']
        bus1 = obj['bus1']['value']
        bus2 = obj['bus2']['value']
        length = float(obj['length']['value'])
        line_config = obj['line_config']['value']
        phase = obj['phase']['value']
        print('line_name: ' + line_name + ', line_config: ' + line_config + ', length: ' + str(length) + ', bus1: ' + bus1 + ', bus2: ' + bus2 + ', phase: ' + phase)

        if line_name not in perLengthImpedenceLinesPos and line_config in Zabc:
            # multiply by scalar length
            lenZabc = Zabc[line_config] * length
            # invert the matrix
            invZabc = np.linalg.inv(lenZabc)
            # test if the inverse * original = identity
            #identityTest = np.dot(lenZabc, invZabc)
            #print('identity test for ' + line_name + ': ' + str(identityTest))
            perLengthImpedenceLinesPos[line_name] = invZabc
            # negate the matrix and assign it back to the dictionary
            perLengthImpedenceLinesNeg[line_name] = invZabc * -1

    for line_name in perLengthImpedenceLinesPos:
        print('perLengthImpedenceLinesPos[' + line_name + ']: ' + str(perLengthImpedenceLinesPos[line_name]))
        print('perLengthImpedenceLinesNeg[' + line_name + ']: ' + str(perLengthImpedenceLinesNeg[line_name]))
    print('')

    print('LINE_MODEL_VALIDATOR DONE!!!', flush=True)
    print('LINE_MODEL_VALIDATOR DONE!!!', file=logfile)

    return


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
    log_file = open('line_model_validator.log', 'w')

    start(log_file, feeder_mrid, model_api_topic)    


if __name__ == "__main__":
    _main()
