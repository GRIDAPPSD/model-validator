# ------------------------------------------------------------------------------
# Copyright (c) 2022, Battelle Memorial Institute All rights reserved.
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
Created on Jan 6, 2022

@author: Gary Black
"""""

import sys
import os
import argparse
import json
import importlib

from gridappsd import GridAPPSD


def fillYbusUnique(bus1, bus2, Ybus):
    if bus1 not in Ybus:
        Ybus[bus1] = {}

    if bus2 in Ybus[bus1]:
        print('    *** WARNING: Unexpected existing value found for Ybus[' + bus1 + '][' + bus2 + '] when filling switching equipment value\n', flush=True)
        print('    *** WARNING: Unexpected existing value found for Ybus[' + bus1 + '][' + bus2 + '] when filling switching equipment value\n', file=logfile)

    # if needed, here's how to find the two immediate calling functions
    #if bus2=='X2673305B.1' and bus1=='X2673305B.2':
    #    print('*** fillYbusUnique bus1: ' + bus1 + ', bus2: ' + bus2 + ', caller: ' + str(inspect.stack()[1].function) + ', ' + str(inspect.stack()[2].function), flush=True)

    Ybus[bus1][bus2] = complex(-500.0, 500.0)


def fillYbusAdd(bus1, bus2, Ybus):
    if bus1 not in Ybus:
        Ybus[bus1] = {}

    if bus2 in Ybus[bus1]:
        Ybus[bus1][bus2] += complex(500.0, -500.0)
    else:
        Ybus[bus1][bus2] = complex(500.0, -500.0)


def fillYbusNoSwap(bus1, bus2, is_Open, Ybus):
    #print('fillYbusNoSwap bus1: ' + bus1 + ', bus2: ' + bus2 + ', is_Open: ' + str(is_Open), flush=True)
    if not is_Open:
        fillYbusUnique(bus2, bus1, Ybus)
        fillYbusAdd(bus1, bus1, Ybus)
        fillYbusAdd(bus2, bus2, Ybus)

def fill_Ybus_SwitchingEquipment_switches(sparql_mgr, Ybus):
    bindings = sparql_mgr.SwitchingEquipment_switch_names()
    #print('SWITCHING_EQUIPMENT_FILL_YBUS switch_names query results:', flush=True)
    #print(bindings, flush=True)

    if len(bindings) == 0:
        return

    # map transformer query phase values to nodelist indexes
    ybusPhaseIdx = {'A': '.1', 'B': '.2', 'C': '.3'}

    for obj in bindings:
        sw_name = obj['sw_name']['value']
        #base_V = int(obj['base_V']['value'])
        is_Open = obj['is_Open']['value'].upper() == 'TRUE'
        #rated_Current = int(obj['rated_Current']['value'])
        #breaking_Capacity = int(obj['breaking_Capacity']['value'])
        #sw_ph_status = obj['sw_ph_status']['value']
        bus1 = obj['bus1']['value'].upper()
        bus2 = obj['bus2']['value'].upper()
        phases_side1 = obj['phases_side1']['value']
        #phases_side2 = obj['phases_side2']['value']
        #print('sw_name: ' + sw_name + ', is_Open: ' + str(is_Open) + ', bus1: ' + bus1 + ', bus2: ' + bus2 + ', phases_side1: (' + phases_side1 + ')' + ', phases_side2: (' + phases_side2 + ')')

        if phases_side1 == '':
            # 3-phase switch
            #print('3-phase switch found bus1: ' + bus1 + ', bus2: ' + bus2, flush=True)
            fillYbusNoSwap(bus1+'.1', bus2+'.1', is_Open, Ybus)
            fillYbusNoSwap(bus1+'.2', bus2+'.2', is_Open, Ybus)
            fillYbusNoSwap(bus1+'.3', bus2+'.3', is_Open, Ybus)

        else:
            # 1- or 2-phase switch
            switchColorIdx = 0
            for phase in phases_side1:
                #print('1/2-phase switch found phase: ' + phase + ', bus1: ' + bus1 + ', bus2: ' + bus2, flush=True)
                if phase in ybusPhaseIdx:
                    fillYbusNoSwap(bus1+ybusPhaseIdx[phase], bus2+ybusPhaseIdx[phase], is_Open, Ybus)


def start(log_file, feeder_mrid, model_api_topic):
    global logfile
    logfile = log_file

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)

    print('\nStarting to build static Ybus...', flush=True)

    Ybus = {}
    Unsupported = {}

    mod_import = importlib.import_module('line_model_validator.line_model_validator')
    start_func = getattr(mod_import, 'start')
    start_func(log_file, feeder_mrid, model_api_topic, False, Ybus, Unsupported)
    #print('line_model_validator static Ybus...')
    #print(Ybus)
    line_count = 0
    for bus1 in Ybus:
        line_count += len(Ybus[bus1])
    print('\nLine_model # entries: ' + str(line_count), flush=True)

    mod_import = importlib.import_module('power_transformer_validator.power_transformer_validator')
    start_func = getattr(mod_import, 'start')
    start_func(log_file, feeder_mrid, model_api_topic, False, Ybus, Unsupported)
    #print('power_transformer_validator static Ybus...')
    #print(Ybus)
    count = 0
    for bus1 in Ybus:
        count += len(Ybus[bus1])
    xfmr_count = count - line_count
    print('\nPower_transformer # entries: ' + str(xfmr_count), flush=True)

    fill_Ybus_SwitchingEquipment_switches(sparql_mgr, Ybus)
    #print('switching_equipment_validator (final) static Ybus...')
    #print(Ybus)
    count = 0
    for bus1 in Ybus:
        count += len(Ybus[bus1])
    switch_count = count - line_count - xfmr_count
    print('\nSwitching_equipment # entries: ' + str(switch_count), flush=True)

    print('\nFull static Ybus:')
    for bus1 in Ybus:
        for bus2 in Ybus[bus1]:
            print(bus1 + ',' + bus2 + ',' + str(Ybus[bus1][bus2].real) + ',' + str(Ybus[bus1][bus2].imag))

    ysysCount = 0
    for bus1 in Ybus:
        ysysCount += len(Ybus[bus1])
    print('\nTotal static Ybus # entries: ' + str(ysysCount) + '\n', flush=True)


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
    log_file = open('static_ybus.log', 'w')

    start(log_file, feeder_mrid, model_api_topic)


if __name__ == "__main__":
    _main()

