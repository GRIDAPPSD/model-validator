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
Created on Apr 29, 2021

@author: Gary Black, Shiva Poudel
"""""

import sys
import os
import argparse
import json
import importlib
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

    #print('\n*** Full Ybus:\n')
    #for bus1 in Ybus:
    #    for bus2 in Ybus[bus1]:
    #        print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag))

    ybusCount = 0
    for bus1 in Ybus:
        ybusCount += len(Ybus[bus1])
    #print('Total Ybus # entries: ' + str(ybusCount) + '\n', flush=True)
    #print('Total Ybus # entries: ' + str(ybusCount) + '\n', file=logfile)

    for bus1 in list(Ybus):
        for bus2 in list(Ybus[bus1]):
            delYFlag = False
            if (bus1 in Ysys) and (bus2 in Ysys[bus1]):
                del Ysys[bus1][bus2]
                if len(Ysys[bus1]) == 0:
                    del Ysys[bus1]

                del Ybus[bus1][bus2]
                if len(Ybus[bus1]) == 0:
                    del Ybus[bus1]
                delYFlag = True

            if (bus2 in Ysys) and (bus1 in Ysys[bus2]):
                del Ysys[bus2][bus1]
                if len(Ysys[bus2]) == 0:
                    del Ysys[bus2]

                if not delYFlag:
                    del Ybus[bus1][bus2]
                    if len(Ybus[bus1]) == 0:
                        del Ybus[bus1]

    redCount = 0
    for bus1 in Ysys:
        redCount += len(Ysys[bus1])
    print('\n*** Missing Ybus entries: ' + str(redCount) + '\n', flush=True)
    print('\n*** Missing Ybus entries: ' + str(redCount) + '\n', file=logfile)

    greenCount = ysysCount - redCount
    yellowCount = 0
    VI = float(greenCount + yellowCount)/float(ysysCount)
    report = []
    report.append([f"Expected entries\N{SUPERSCRIPT ONE}", ysysCount, "{:.4f}".format(VI), greenCount, yellowCount, redCount])

    for bus1 in Ysys:
        for bus2 in Ysys[bus1]:
            print(bus1 + ',' + bus2 + ',' + str(Ysys[bus1][bus2].real) + ',' + str(Ysys[bus1][bus2].imag) + ',' + redCircle(True), flush=True)
            print(bus1 + ',' + bus2 + ',' + str(Ysys[bus1][bus2].real) + ',' + str(Ysys[bus1][bus2].imag) + ',' + redCircle(False), file=logfile)

    unexpectedCount = 0
    for bus1 in Ybus:
        unexpectedCount += len(Ybus[bus1])
    print('\n*** Unexpected Ybus entries: ' + str(unexpectedCount) + '\n', flush=True)
    print('\n*** Unexpected Ybus entries: ' + str(unexpectedCount) + '\n', file=logfile)

    yellowCount = 0
    redCount = 0
    for bus1 in Ybus:
        for bus2 in Ybus[bus1]:
            if abs(Ybus[bus1][bus2] - complex(0.0, 0.0)) > 1.0e-9:
                short_bus1 = bus1.split('.')[0]
                short_bus2 = bus2.split('.')[0]
                if short_bus1 in Unsupported and short_bus2 in Unsupported[short_bus1][0]:
                    print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag) + ',' + yellowCircle(True) + ' ,***UNSUPPORTED: ' + Unsupported[short_bus1][1], flush=True)
                    print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag) + ',' + yellowCircle(False) + ' ,***UNSUPPORTED: ' + Unsupported[short_bus1][1], file=logfile)
                    yellowCount += 1
                else:
                    print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag) + ',' + redCircle(True), flush=True)
                    print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag) + ',' + redCircle(False), file=logfile)
                    redCount += 1
            else:
                print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag) + ',' + yellowCircle(True) + ' ,***NEAR_ZERO', flush=True)
                print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag) + ',' + yellowCircle(False) + ' ,***NEAR_ZERO', file=logfile)
                yellowCount += 1

    greenCount = ybusCount - unexpectedCount
    VI = float(ybusCount - redCount)/float(ybusCount)
    report.append([f"Existing entries\N{SUPERSCRIPT TWO}", ybusCount, "{:.4f}".format(VI), greenCount, yellowCount, redCount])

    print('\n', flush=True)
    print(tabulate(report, headers=["Ybus check", "Entries checked", "VI", greenCircle(True), yellowCircle(True), redCircle(True)], tablefmt="fancy_grid"), flush=True)
    print('', flush=True)

    print('\n', file=logfile)
    print(tabulate(report, headers=["Ybus check", "Entries checked", "VI", greenCircle(False), yellowCircle(False), redCircle(False)], tablefmt="fancy_grid"), file=logfile)
    print('', file=logfile)

    print(f"\N{SUPERSCRIPT ONE}Checks whether each expected entry is found in Ybus where green=found; yellow=not found, but explainable; red=not found", flush=True)
    print(f"\N{SUPERSCRIPT TWO}Checks whether each existing entry in Ybus is expected where green=expected; yellow=unexpected, but explainable; red=unexpected\n", flush=True)
    print(f"\N{SUPERSCRIPT ONE}Checks whether each expected entry is found in Ybus where green=found; yellow=not found, but explainable; red=not found", file=logfile)
    print(f"\N{SUPERSCRIPT TWO}Checks whether each existing entry in Ybus is expected where green=expected; yellow=unexpected, but explainable; red=unexpected\n", file=logfile)


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

