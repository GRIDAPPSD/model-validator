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

from gridappsd import GridAPPSD


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
    log_file = open('load_flow_validator.log', 'w')

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

    mod_import = importlib.import_module('line_model_validator.line_model_validator')
    start_func = getattr(mod_import, 'start')
    start_func(log_file, feeder_mrid, model_api_topic, False, Ysys)
    #print('line_model_validator Ysys...')
    #print(Ysys)
    print('line_model # bus1 items: ' + str(len(Ysys)))
    count = 0
    for bus1 in Ysys:
        count += len(Ysys[bus1])
    print('line_model # bus1+bus2 values: ' + str(count) + '\n')

    mod_import = importlib.import_module('power_transformer_validator.power_transformer_validator')
    start_func = getattr(mod_import, 'start')
    start_func(log_file, feeder_mrid, model_api_topic, False, Ysys)
    #print('power_transformer_validator Ysys...')
    #print(Ysys)
    print('power_transformer # bus1 items: ' + str(len(Ysys)))
    count = 0
    for bus1 in Ysys:
        count += len(Ysys[bus1])
    print('power_transformer # bus1+bus2 values: ' + str(count) + '\n')

    mod_import = importlib.import_module('switching_equipment_validator.switching_equipment_validator')
    start_func = getattr(mod_import, 'start')
    start_func(log_file, feeder_mrid, model_api_topic, False, Ysys)
    #print('switching_equipment_validator (final) Ysys...')
    #print(Ysys)
    print('switching_equipment # bus1 items: ' + str(len(Ysys)))
    count = 0
    for bus1 in Ysys:
        count += len(Ysys[bus1])
    print('switching_equipment # bus1+bus2 values: ' + str(count) + '\n')

    #print('\n***** Full Ysys:\n')
    #for bus1 in Ysys:
    #    for bus2 in Ysys[bus1]:
    #        print(bus1 + ',' + bus2 + ',' + str(Ysys[bus1][bus2].real) + ',' + str(Ysys[bus1][bus2].imag))

    print('Full Ysys #bus1 items: ' + str(len(Ysys)))
    count = 0
    for bus1 in Ysys:
        count += len(Ysys[bus1])
    print('Full Ysys total items: ' + str(count) + '\n')

    #print('\n***** Full Ybus:\n')
    #for bus1 in Ybus:
    #    for bus2 in Ybus[bus1]:
    #        print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag))

    print('Full Ybus #bus1 items: ' + str(len(Ysys)))
    count = 0
    for bus1 in Ybus:
        count += len(Ybus[bus1])
    print('Full Ybus total items: ' + str(count) + '\n')

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

    print('\n***** Unmatched in Ysys:\n')
    for bus1 in Ysys:
        for bus2 in Ysys[bus1]:
            print(bus1 + ',' + bus2 + ',' + str(Ysys[bus1][bus2].real) + ',' + str(Ysys[bus1][bus2].imag))

    print('Unmatched in Ysys #bus1 items: ' + str(len(Ysys)))
    count = 0
    for bus1 in Ysys:
        count += len(Ysys[bus1])
    print('Unmatched in Ysys total items: ' + str(count) + '\n')

    print('\n***** Unmatched in Ybus:\n')
    for bus1 in Ybus:
        for bus2 in Ybus[bus1]:
            print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag))

    print('Unmatched in Ybus #bus1 items: ' + str(len(Ybus)))
    count = 0
    for bus1 in Ybus:
        count += len(Ybus[bus1])
    print('Unmatched in Ybus total items: ' + str(count) + '\n')


if __name__ == "__main__":
    _main()

