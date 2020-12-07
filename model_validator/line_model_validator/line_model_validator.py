# -------------------------------------------------------------------------------
# Copyright (c) 2020, Battelle Memorial Institute All rights reserved.
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


def diffColor(diffValue):
    if diffValue < 1e-3:
        return 'GREEN'
    elif diffValue > 1e-2:
        return 'RED'
    else:
        return 'YELLOW'


def diffPercentReal(YprimValue, YbusValue):
    global minPercentDiffReal, maxPercentDiffReal

    ratio = YprimValue/YbusValue

    if ratio > 1.0:
        percent = 100.0*(ratio - 1.0)
    else:
        percent = 100.0*(1.0 - ratio)

    minPercentDiffReal = min(minPercentDiffReal, percent)
    maxPercentDiffReal = max(maxPercentDiffReal, percent)

    return percent


def diffPercentImag(YprimValue, YbusValue):
    global minPercentDiffImag, maxPercentDiffImag

    ratio = YprimValue/YbusValue

    if ratio > 1.0:
        percent = 100.0*(ratio - 1.0)
    else:
        percent = 100.0*(1.0 - ratio)

    minPercentDiffImag = min(minPercentDiffImag, percent)
    maxPercentDiffImag = max(maxPercentDiffImag, percent)

    return percent


def compareY(line_name, pairA, pairB, YprimValue, Ybus):
    if pairA in Ybus and pairB in Ybus[pairA]:
        row = pairA
        col = pairB
    elif pairB in Ybus and pairA in Ybus[pairB]:
        row = pairB
        col = pairA
    else:
        print('*** ERROR: Ybus match NOT FOUND for Ybus[' + pairA + '][' + pairB + ']', flush=True)
        print('*** ERROR: Ybus match NOT FOUND for Ybus[' + pairA + '][' + pairB + ']', file=logfile)
        return

    YbusValue = Ybus[row][col]

    print("    between i: " + row + ", and j: " + col, flush=True)
    print("    between i: " + row + ", and j: " + col, file=logfile)

    realDiff = abs(YprimValue.real - YbusValue.real)
    realPer = diffPercentReal(YprimValue.real, YbusValue.real)
    print("        Real Ybus[i,j]:" + "{:10.6f}".format(YbusValue.real) + ", computed:" + "{:10.6f}".format(YprimValue.real) + ", % diff:" + "{:10.6f}".format(realPer) + ", " + diffColor(realDiff), flush=True)
    print("        Real Ybus[i,j]:" + "{:10.6f}".format(YbusValue.real) + ", computed:" + "{:10.6f}".format(YprimValue.real) + ", % diff:" + "{:10.6f}".format(realPer) + ", " + diffColor(realDiff), file=logfile)

    imagDiff = abs(YprimValue.imag - YbusValue.imag)
    imagPer = diffPercentImag(YprimValue.imag, YbusValue.imag)
    print("        Imag Ybus[i,j]:" + "{:10.6f}".format(YbusValue.imag) + ", computed:" + "{:10.6f}".format(YprimValue.imag) + ", % diff:" + "{:10.6f}".format(imagPer) + ", " + diffColor(imagDiff), flush=True)
    print("        Imag Ybus[i,j]:" + "{:10.6f}".format(YbusValue.imag) + ", computed:" + "{:10.6f}".format(YprimValue.imag) + ", % diff:" + "{:10.6f}".format(imagPer) + ", " + diffColor(imagDiff), file=logfile)


def check_perLengthImpedence_lines(sparql_mgr, Ybus):
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

    #for line_config in Zabc:
    #    print('Zabc[' + line_config + ']: ' + str(Zabc[line_config]))
    #print('')

    bindings = sparql_mgr.perLengthImpedence_line_names()
    #print('LINE_MODEL_VALIDATOR line_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('LINE_MODEL_VALIDATOR line_names query results:', file=logfile)
    #print(bindings, file=logfile)

    ybusPhaseIdx = {'A': '.1', 'B': '.2', 'C': '.3'}
    yprimPhaseIdx = {'A': 0, 'B': 1, 'C': 2}

    global minPercentDiffReal, maxPercentDiffReal
    minPercentDiffReal = 100.0
    maxPercentDiffReal = 0.0
    global minPercentDiffImag, maxPercentDiffImag
    minPercentDiffImag = 100.0
    maxPercentDiffImag = 0.0

    last_name = ''
    for obj in bindings:
        line_name = obj['line_name']['value']
        bus1 = obj['bus1']['value'].upper()
        bus2 = obj['bus2']['value'].upper()
        length = float(obj['length']['value'])
        line_config = obj['line_config']['value']

        if 'phase' not in obj:
            print("*** ERROR: No phase for line_name: " + line_name, flush=True)
            print("*** ERROR: No phase for line_name: " + line_name, file=logfile)
            continue
        phase = obj['phase']['value']
        if phase!='A' and phase!='B' and phase!='C':
            print("*** ERROR: Phase value other than A,B,C for line_name: " + line_name + ", phase: " + phase, flush=True)
            print("*** ERROR: Phase value other than A,B,C for line_name: " + line_name + ", phase: " + phase, file=logfile)
            continue
        #print('line_name: ' + line_name + ', line_config: ' + line_config + ', length: ' + str(length) + ', bus1: ' + bus1 + ', bus2: ' + bus2 + ', phase: ' + phase)

        if line_name!=last_name and line_config in Zabc:
            print("\nValidating line_name: " + line_name, flush=True)
            print("\nValidating line_name: " + line_name, file=logfile)

            last_name = line_name
            line_idx = 0

            # multiply by scalar length
            lenZabc = Zabc[line_config] * length
            # invert the matrix
            invZabc = np.linalg.inv(lenZabc)
            # test if the inverse * original = identity
            #identityTest = np.dot(lenZabc, invZabc)
            #print('identity test for ' + line_name + ': ' + str(identityTest))
            # negate the matrix and assign it to Yprim
            Yprim = invZabc * -1

        # we now have the negated inverted matrix for comparison
        ybusIdx = ybusPhaseIdx[phase]
        pairA = bus1 + ybusIdx
        pairB = bus2 + ybusIdx
        line_idx += 1

        if Yprim.size == 1:
            # do comparisons now
            compareY(line_name, pairA, pairB, Yprim[0,0], Ybus)

        elif Yprim.size == 4:
            if line_idx == 1:
                pair1A = pairA
                pair1B = pairB
            else:
                pair2A = pairA
                pair2B = pairB

                # do comparisons now
                compareY(line_name, pair1A, pair1B, Yprim[0,0], Ybus)
                compareY(line_name, pair2A, pair1B, Yprim[1,0], Ybus)
                compareY(line_name, pair2A, pair2B, Yprim[1,1], Ybus)

        elif Yprim.size == 9:
            if line_idx == 1:
                pair1A = pairA
                pair1B = pairB
            elif line_idx == 2:
                pair2A = pairA
                pair2B = pairB
            else:
                pair3A = pairA
                pair3B = pairB

                # do comparisons now
                compareY(line_name, pair1A, pair1B, Yprim[0,0], Ybus)
                compareY(line_name, pair2A, pair1B, Yprim[1,0], Ybus)
                compareY(line_name, pair2A, pair2B, Yprim[1,1], Ybus)
                compareY(line_name, pair3A, pair1B, Yprim[2,0], Ybus)
                compareY(line_name, pair3A, pair2B, Yprim[2,1], Ybus)
                compareY(line_name, pair3A, pair3B, Yprim[2,2], Ybus)

    print("\nReal minimum % difference:" + "{:10.6f}".format(minPercentDiffReal), flush=True)
    print("\nReal minimum % difference:" + "{:10.6f}".format(minPercentDiffReal), file=logfile)
    print("Real maximum % difference:" + "{:10.6f}".format(maxPercentDiffReal), flush=True)
    print("Real maximum % difference:" + "{:10.6f}".format(maxPercentDiffReal), file=logfile)

    print("\nImag minimum % difference:" + "{:10.6f}".format(minPercentDiffImag), flush=True)
    print("\nImag minimum % difference:" + "{:10.6f}".format(minPercentDiffImag), file=logfile)
    print("Imag maximum % difference:" + "{:10.6f}".format(maxPercentDiffImag), flush=True)
    print("Imag maximum % difference:" + "{:10.6f}".format(maxPercentDiffImag), file=logfile)

    return


def start(log_file, feeder_mrid, model_api_topic):
    global logfile
    logfile = log_file

    print("\nLINE_MODEL_VALIDATOR starting!!!----------------------------------------------------")
    print("\nLINE_MODEL_VALIDATOR starting!!!----------------------------------------------------", file=logfile)

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)

    ysparse,nodelist = sparql_mgr.ybus_export()

    idx = 1
    nodes = {}
    for obj in nodelist:
        nodes[idx] = obj.strip('\"')
        idx += 1
    #print(nodes)

    Ybus = {}
    for obj in ysparse:
        items = obj.split(',')
        if items[0] == 'Row':
            continue
        if nodes[int(items[0])] not in Ybus:
            Ybus[nodes[int(items[0])]] = {}
        Ybus[nodes[int(items[0])]][nodes[int(items[1])]] = complex(float(items[2]), float(items[3]))
    #print(Ybus)

    check_perLengthImpedence_lines(sparql_mgr, Ybus)

    print('\nLINE_MODEL_VALIDATOR DONE!!!', flush=True)
    print('\nLINE_MODEL_VALIDATOR DONE!!!', file=logfile)

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
