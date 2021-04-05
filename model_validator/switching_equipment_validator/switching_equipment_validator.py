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
Created on Feb 19, 2021

@author: Gary Black, Shiva Poudel
"""""

#from shared.sparql import SPARQLManager

import math
import argparse
import json
import sys
import os
import importlib
import numpy as np
from tabulate import tabulate

from gridappsd import GridAPPSD

global logfile


def diffColor(colorIdx, colorFlag):
    if colorIdx == 0:
        return '\u001b[32m\u25cf\u001b[37m' if colorFlag else '\u25cb'
    elif colorIdx == 1:
        return '\u001b[33m\u25cf\u001b[37m' if colorFlag else '\u25d1'
    else:
        return '\u001b[31m\u25cf\u001b[37m' if colorFlag else '\u25cf'


def compareY(is_Open, pair_b1, pair_b2, Ybus):
    global greenCountReal, yellowCountReal, redCountReal
    global greenCountImag, yellowCountImag, redCountImag
    global greenCount, yellowCount, redCount

    noEntryFlag = False
    if pair_b1 in Ybus and pair_b2 in Ybus[pair_b1]:
        row = pair_b1
        col = pair_b2
        YbusValue = Ybus[row][col]
    elif pair_b2 in Ybus and pair_b1 in Ybus[pair_b2]:
        row = pair_b2
        col = pair_b1
        YbusValue = Ybus[row][col]
    else:
        row = pair_b1
        col = pair_b2
        YbusValue = complex(0.0, 0.0)
        noEntryFlag = True

    print("    between i: " + row + ", and j: " + col, flush=True)
    print("    between i: " + row + ", and j: " + col, file=logfile)

    # don't flag missing Ybus entries for open switches because that's expected
    if noEntryFlag and not is_Open:
        print('        *** WARNING: Entry NOT FOUND for Ybus[' + row + '][' + col + ']', flush=True)
        print('        *** WARNING: Entry NOT FOUND for Ybus[' + row + '][' + col + ']', file=logfile)

    if is_Open:
        if abs(YbusValue.real) <= 0.001:
            realColorIdx = 0
            greenCountReal += 1
        else:
            realColorIdx = 2
            redCountReal += 1
    else:
        if YbusValue.real>=-1000.0 and YbusValue.real<=-500.0:
            realColorIdx = 0
            greenCountReal += 1
        elif abs(YbusValue.real) <= 0.001:
            realColorIdx = 2
            redCountReal += 1
        else:
            realColorIdx = 1
            yellowCountReal += 1

    print("        Real Ybus[i,j]:" + "{:13.6f}".format(YbusValue.real) + "  " + diffColor(realColorIdx, True), flush=True)
    print("        Real Ybus[i,j]:" + "{:13.6f}".format(YbusValue.real) + "  " + diffColor(realColorIdx, False), file=logfile)

    if is_Open:
        if abs(YbusValue.imag) <= 0.001:
            imagColorIdx = 0
            greenCountImag += 1
        else:
            imagColorIdx = 2
            redCountImag += 1
    else:
        if YbusValue.imag>=500.0 and YbusValue.imag<=1000.0:
            imagColorIdx = 0
            greenCountImag += 1
        elif abs(YbusValue.imag) <= 0.001:
            imagColorIdx = 2
            redCountImag += 1
        else:
            imagColorIdx = 1
            yellowCountImag += 1

    print("        Imag Ybus[i,j]:" + "{:13.6f}".format(YbusValue.imag) + "  " + diffColor(imagColorIdx, True), flush=True)
    print("        Imag Ybus[i,j]:" + "{:13.6f}".format(YbusValue.imag) + "  " + diffColor(imagColorIdx, False), file=logfile)

    return max(realColorIdx, imagColorIdx)


def fillYsys(is_Open, bus1, bus2, Ysys):
    if bus1 not in Ysys:
        Ysys[bus1] = {}

    if bus2 in Ysys[bus1]:
        print('    *** WARNING: Unexpected existing value found for Ysys[' + bus1 + '][' + bus2 + '] when filling switching equipment value\n', flush=True)
        print('    *** WARNING: Unexpected existing value found for Ysys[' + bus1 + '][' + bus2 + '] when filling switching equipment value\n', file=logfile)

    if is_Open:
        Ysys[bus1][bus2] = complex(0.0, 0.0)
    else:
        Ysys[bus1][bus2] = complex(-500.0, 500.0)


def validate_SwitchingEquipment_switches(sparql_mgr, Ybus, cmpFlag, Ysys):
    if cmpFlag:
        print('\nSWITCHING_EQUIPMENT_VALIDATOR switches validation...\n', flush=True)
        print('\nSWITCHING_EQUIPMENT_VALIDATOR switches validation...\n', file=logfile)

    # return # of switches validated
    switches_count = 0

    bindings = sparql_mgr.SwitchingEquipment_switch_names()
    #print('SWITCHING_EQUIPMENT_VALIDATOR switch_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('SWITCHING_EQUIPMENT_VALIDATOR switch_names query results:', file=logfile)
    #print(bindings, file=logfile)

    if len(bindings) == 0:
        if cmpFlag:
            print('\nSWITCHING_EQUIPMENT_VALIDATOR switches: NO SWITCH MATCHES', flush=True)
            print('\nSWITCHING_EQUIPMENT_VALIDATOR switches: NO SWITCH MATCHES', file=logfile)
        return switches_count

    if cmpFlag:
        global greenCountReal, yellowCountReal, redCountReal
        greenCountReal = yellowCountReal = redCountReal = 0
        global greenCountImag, yellowCountImag, redCountImag
        greenCountImag = yellowCountImag = redCountImag = 0
        global greenCount, yellowCount, redCount
        greenCount = yellowCount = redCount = 0

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

        if cmpFlag:
            print('Validating switch_name: ' + sw_name, flush=True)
            print('Validating switch_name: ' + sw_name, file=logfile)

        if phases_side1 == '':
            # 3-phase switch
            if cmpFlag:
                colorIdx11 = compareY(is_Open, bus1+'.1', bus2+'.1', Ybus)
                colorIdx22 = compareY(is_Open, bus1+'.2', bus2+'.2', Ybus)
                colorIdx33 = compareY(is_Open, bus1+'.3', bus2+'.3', Ybus)
                switchColorIdx = max(colorIdx11, colorIdx22, colorIdx33)
            else:
                fillYsys(is_Open, bus1+'.1', bus2+'.1', Ysys)
                fillYsys(is_Open, bus1+'.2', bus2+'.2', Ysys)
                fillYsys(is_Open, bus1+'.3', bus2+'.3', Ysys)

        else:
            # 1- or 2-phase switch
            switchColorIdx = 0
            for phase in phases_side1:
                if phase in ybusPhaseIdx:
                    if cmpFlag:
                        colorIdx = compareY(is_Open, bus1+ybusPhaseIdx[phase], bus2+ybusPhaseIdx[phase], Ybus)
                        switchColorIdx = max(switchColorIdx, colorIdx)
                    else:
                        fillYsys(is_Open, bus1+ybusPhaseIdx[phase], bus2+ybusPhaseIdx[phase], Ysys)
                elif cmpFlag:
                    print('    *** WARNING: switch phase other than A, B, or C found, ' + phases_side1 + ', for switch : ' + sw_name + '\n', flush=True)
                    print('    *** WARNING: switch phase other than A, B, or C found, ' + phases_side1 + ', for switch : ' + sw_name + '\n', file=logfile)

        switches_count += 1

        if cmpFlag:
            if switchColorIdx == 0:
                greenCount += 1
            elif switchColorIdx == 1:
                yellowCount += 1
            else:
                redCount += 1

            print("", flush=True)
            print("", file=logfile)

    if cmpFlag:
        print("\nSummary for SwitchingEquipment switches:", flush=True)
        print("\nSummary for SwitchingEquipment switches:", file=logfile)

        print("\nReal \u001b[32m\u25cf\u001b[37m  count: " + str(greenCountReal), flush=True)
        print("\nReal \u25cb  count: " + str(greenCountReal), file=logfile)
        print("Real \u001b[33m\u25cf\u001b[37m  count: " + str(yellowCountReal), flush=True)
        print("Real \u25d1  count: " + str(yellowCountReal), file=logfile)
        print("Real \u001b[31m\u25cf\u001b[37m  count: " + str(redCountReal), flush=True)
        print("Real \u25cf  count: " + str(redCountReal), file=logfile)

        print("\nImag \u001b[32m\u25cf\u001b[37m  count: " + str(greenCountImag), flush=True)
        print("\nImag \u25cb  count: " + str(greenCountImag), file=logfile)
        print("Imag \u001b[33m\u25cf\u001b[37m  count: " + str(yellowCountImag), flush=True)
        print("Imag \u25d1  count: " + str(yellowCountImag), file=logfile)
        print("Imag \u001b[31m\u25cf\u001b[37m  count: " + str(redCountImag), flush=True)
        print("Imag \u25cf  count: " + str(redCountImag), file=logfile)

        print("\nFinished validation for SwitchingEquipment switches", flush=True)
        print("\nFinished validation for SwitchingEquipment switches", file=logfile)

    return switches_count


def start(log_file, feeder_mrid, model_api_topic, cmpFlag=True, Ysys=None):
    global logfile
    logfile = log_file

    if cmpFlag:
        print("\nSWITCHING_EQUIPMENT_VALIDATOR starting!!!-----------------------------------------")
        print("\nSWITCHING_EQUIPMENT_VALIDATOR starting!!!-----------------------------------------", file=logfile)

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic)

    if cmpFlag:
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

        # list of lists for the tabular report
        report = []
    else:
        Ybus = None

    SwitchingEquipment_switches = validate_SwitchingEquipment_switches(sparql_mgr, Ybus, cmpFlag, Ysys)
    if cmpFlag:
        if SwitchingEquipment_switches > 0:
            count = greenCount + yellowCount + redCount
            VI = float(count - redCount)/float(count)
            report.append([SwitchingEquipment_switches, "{:.4f}".format(VI), greenCount, yellowCount, redCount])
        else:
            report.append([SwitchingEquipment_switches])

        print('\n', flush=True)
        print(tabulate(report, headers=["# Switches", "VI", diffColor(0, True), diffColor(1, True), diffColor(2, True)], tablefmt="fancy_grid"), flush=True)
        print('\n', file=logfile)
        print(tabulate(report, headers=["# Switches", "VI", diffColor(0, False), diffColor(1, False), diffColor(2, False)], tablefmt="fancy_grid"), file=logfile)

        print('\nSWITCHING_EQUIPMENT_VALIDATOR DONE!!!', flush=True)
        print('\nSWITCHING_EQUIPMENT_VALIDATOR DONE!!!', file=logfile)


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
    log_file = open('switching_equipment_validator.log', 'w')

    start(log_file, feeder_mrid, model_api_topic)


if __name__ == "__main__":
    _main()
