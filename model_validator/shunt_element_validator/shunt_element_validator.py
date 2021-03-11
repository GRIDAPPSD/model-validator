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
Created on March 2, 2021

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


def diffColorIdxImag(absDiff):
    global greenCountImag, yellowCountImag, redCountImag

    if absDiff < 1e-3:
        greenCountImag += 1
        return 0
    elif absDiff >= 1e-2:
        redCountImag += 1
        return 2
    else:
        yellowCountImag += 1
        return 1


def diffColorIdxReal(absDiff):
    global greenCountReal, yellowCountReal, redCountReal

    if absDiff < 1e-3:
        greenCountReal += 1
        return 0
    elif absDiff >= 1e-2:
        redCountReal += 1
        return 2
    else:
        yellowCountReal += 1
        return 1


def compareShuntImag(sum_shunt_imag, Yshunt_imag):
    absDiff = abs(sum_shunt_imag - Yshunt_imag)
    colorIdx = diffColorIdxImag(absDiff)
    print("    Imag shunt element total:" + "{:12.6g}".format(sum_shunt_imag) + ", computed Yshunt:" + "{:12.6g}".format(Yshunt_imag) + "  " + diffColor(colorIdx, True), flush=True)
    print("    Imag shunt element total:" + "{:12.6g}".format(sum_shunt_imag) + ", computed Yshunt:" + "{:12.6g}".format(Yshunt_imag) + "  " + diffColor(colorIdx, False), file=logfile)

    return colorIdx


def compareShuntReal(sum_shunt_real, Yshunt_real):
    absDiff = abs(sum_shunt_real - Yshunt_real)
    colorIdx = diffColorIdxReal(absDiff)
    print("    Real shunt element total:" + "{:12.6g}".format(sum_shunt_real) + ", computed Yshunt:" + "{:12.6g}".format(Yshunt_real) + "  " + diffColor(colorIdx, True), flush=True)
    print("    Real shunt element total:" + "{:12.6g}".format(sum_shunt_real) + ", computed Yshunt:" + "{:12.6g}".format(Yshunt_real) + "  " + diffColor(colorIdx, False), file=logfile)

    return colorIdx


def validate_ShuntElement_elements(sparql_mgr, Ybus, Yexp, CNV):
    # map query phase values to nodelist indexes
    ybusPhaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 's1': '.1', 's2': '.2'}

    # CAPACITORS DATA STRUCTURES INITIALIZATION
    bindings = sparql_mgr.ShuntElement_cap_names()
    #print('SHUNT_ELEMENT_VALIDATOR ShuntElement cap_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR ShuntElement cap_names query results:', file=logfile)
    #print(bindings, file=logfile)

    Cap_name = {}
    B_per_section = {}
    for obj in bindings:
        cap_name = obj['cap_name']['value']
        B_per_section[cap_name] = float(obj['b_per_section']['value'])
        bus = obj['bus']['value'].upper()
        phase = 'ABC' # no phase specified indicates 3-phase
        if 'phase' in obj:
            phase = obj['phase']['value']
        mode = 'voltage'
        if 'mode' in obj:
            mode = obj['mode']['value']
        #print('cap_name: ' + cap_name + ', b_per_section: ' + str(b_per_section) + ', phase: ' + phase + ', mode: ' + mode)

        if mode != 'timeScheduled':
            if phase == 'ABC': # 3-phase
                if bus+'.1' not in Cap_name:
                    Cap_name[bus+'.1'] = []
                    Cap_name[bus+'.2'] = []
                    Cap_name[bus+'.3'] = []
                Cap_name[bus+'.1'].append(cap_name)
                Cap_name[bus+'.2'].append(cap_name)
                Cap_name[bus+'.3'].append(cap_name)
            else: # specified phase only
                if bus+ybusPhaseIdx[phase] not in Cap_name:
                    Cap_name[bus+ybusPhaseIdx[phase]] = []
                Cap_name[bus+ybusPhaseIdx[phase]].append(cap_name)

    # TRANSFORMERS DATA STRUCTURES INITIALIZATION
    bindings = sparql_mgr.TransformerTank_xfmr_rated()
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_rated query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_rated query results:', file=logfile)
    #print(bindings, file=logfile)

    # TransformerTank queries
    RatedS_tank = {}
    RatedU_tank = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        enum = int(obj['enum']['value'])
        if xfmr_name not in RatedS_tank:
            RatedS_tank[xfmr_name] = {}
            RatedU_tank[xfmr_name] = {}

        RatedS_tank[xfmr_name][enum] = int(obj['ratedS']['value'])
        RatedU_tank[xfmr_name][enum] = int(obj['ratedU']['value'])
        #print('xfmr_name: ' + xfmr_name + ', enum: ' + str(enum) + ', ratedS: ' + str(RatedS_tank[xfmr_name][enum]) + ', ratedU: ' + str(RatedU_tank[xfmr_name][enum]))

    bindings = sparql_mgr.TransformerTank_xfmr_nlt()
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_nlt query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_nlt query results:', file=logfile)
    #print(bindings, file=logfile)

    Noloadloss = {}
    I_exciting = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        Noloadloss[xfmr_name] = float(obj['noloadloss_kW']['value'])
        I_exciting[xfmr_name] = float(obj['i_exciting']['value'])
        #print('xfmr_name: ' + xfmr_name + ', noloadloss: ' + str(Noloadloss[xfmr_name]) + ', i_exciting: ' + str(I_exciting[xfmr_name]))

    bindings = sparql_mgr.TransformerTank_xfmr_names()
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_names query results:', file=logfile)
    #print(bindings, file=logfile)

    Xfmr_tank_name = {}
    Enum_tank = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        enum = int(obj['enum']['value'])
        bus = obj['bus']['value'].upper()
        if xfmr_name not in Enum_tank:
            Enum_tank[xfmr_name] = {}

        Enum_tank[xfmr_name][bus] = enum

        phase = obj['phase']['value']
        if phase == 'ABC':
            if bus+'.1' not in Xfmr_tank_name:
                Xfmr_tank_name[bus+'.1'] = []
                Xfmr_tank_name[bus+'.2'] = []
                Xfmr_tank_name[bus+'.3'] = []
            Xfmr_tank_name[bus+'.1'].append(xfmr_name)
            Xfmr_tank_name[bus+'.2'].append(xfmr_name)
            Xfmr_tank_name[bus+'.3'].append(xfmr_name)
        else:
            if bus+ybusPhaseIdx[phase] not in Xfmr_tank_name:
                Xfmr_tank_name[bus+ybusPhaseIdx[phase]] = []
            Xfmr_tank_name[bus+ybusPhaseIdx[phase]].append(xfmr_name)

        #print('xfmr_tank_name: ' + xfmr_name + ', bus: ' + bus + ', phase: ' + phase)

    # TransformerEnd queries
    bindings = sparql_mgr.PowerTransformerEnd_xfmr_admittances()
    #print('SHUNT_ELEMENT_VALIDATOR PowerTransformerEnd xfmr_admittances query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR PowerTransformerEnd xfmr_admittances query results:', file=logfile)
    #print(bindings, file=logfile)

    B_S = {}
    G_S = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        B_S[xfmr_name] = float(obj['b_S']['value'])
        G_S[xfmr_name] = float(obj['g_S']['value'])
        #print('xfmr_name: ' + xfmr_name + ', b_S: ' + str(B_S[xfmr_name]) + ', g_S: ' + str(G_S[xfmr_name])

    bindings = sparql_mgr.PowerTransformerEnd_xfmr_names()
    #print('SHUNT_ELEMENT_VALIDATOR PowerTransformerEnd xfmr_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR PowerTransformerEnd xfmr_names query results:', file=logfile)
    #print(bindings, file=logfile)

    Xfmr_end_name = {}
    RatedU_end = {}
    Enum_end = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        enum = int(obj['end_number']['value'])
        bus = obj['bus']['value'].upper()
        if bus+'.1' not in Xfmr_end_name:
            Xfmr_end_name[bus+'.1'] = []
            Xfmr_end_name[bus+'.2'] = []
            Xfmr_end_name[bus+'.3'] = []
        Xfmr_end_name[bus+'.1'].append(xfmr_name)
        Xfmr_end_name[bus+'.2'].append(xfmr_name)
        Xfmr_end_name[bus+'.3'].append(xfmr_name)

        if xfmr_name not in Enum_end:
            Enum_end[xfmr_name] = {}
            RatedU_end[xfmr_name] = {}

        Enum_end[xfmr_name][bus] = enum
        RatedU_end[xfmr_name][enum] = int(obj['ratedU']['value'])
        #print('xfmr_end_name: ' + xfmr_name + ', end_number: ' + str(enum) + ', bus: ' + bus + ', ratedU: ' + str(RatedU_end[xfmr_name][enum]))

    # Just for checking how often we encounter the more exotic cases
    MultiElem = {}
    MultiCap = {}
    MultiXfmrTank = {}
    MultiXfmrEnd = {}

    # Final validation -- check all nodes for shunt elements
    for node1 in CNV:
        numsum = complex(0.0, 0.0)
        #print('finding shunt_adm for node: ' + node1)
        for node2 in Yexp[node1]:
            #print('\tforward summing connection to node: ' + node2)
            numsum += Yexp[node1][node2]*CNV[node2]
        for node2 in Ybus[node1]:
            if node2 != node1:
                #print('\tbackward summing connection to node: ' + node2)
                numsum += Ybus[node1][node2]*CNV[node2]

        Yshunt = numsum/CNV[node1]

        print('\nValidating shunt element node: ' + node1, flush=True)
        print('\nValidating shunt element node: ' + node1, file=logfile)

        print('cnv(' + node1 + ') = ' + str(CNV[node1]))
        print('Yshunt(' + node1 + ') = ' + str(Yshunt))

        # sum over all capacitors and transformers for their contribution
        # to the total shunt admittance to compare with the computed Yshunt
        sum_shunt_imag = sum_shunt_real = 0.0
        num_elem = 0

        # add in capacitor contribution if applicable
        if node1 in Cap_name:
            for cap in Cap_name[node1]:
                num_elem += 1
                sum_shunt_imag += B_per_section[cap]
                print('Adding capacitor imag contribution: ' + str(B_per_section[cap]))
                # capacitors only contribute to the imaginary part
            if len(Cap_name[node1]) > 1:
                MultiCap[node1] = len(Cap_name[node1])

        # strip phase off node to find bus as this is needed for transformers
        bus = node1.split('.')[0]

        # add in TransformerTank transformer contribution if applicable
        if node1 in Xfmr_tank_name:
            for xfmr in Xfmr_tank_name[node1]:
                print('Checking tank transformer name: ' + xfmr +', enum: ' + str(Enum_tank[xfmr]))
                if Enum_tank[xfmr][bus] == 2:
                    num_elem += 1
                    ratedU_sq = RatedU_tank[xfmr][2]*RatedU_tank[xfmr][2]
                    zBaseS = ratedU_sq/RatedS_tank[xfmr][2]
                    sum_shunt_imag += -I_exciting[xfmr]/(100.0*zBaseS)
                    print('Adding tank transformer imag contribution: ' + str(-I_exciting[xfmr]/(100.0*zBaseS)))
                    sum_shunt_real += (Noloadloss[xfmr]*1000.0)/ratedU_sq
                    print('Adding tank transformer real contribution: ' + str((Noloadloss[xfmr]*1000.0)/ratedU_sq))
            if len(Xfmr_tank_name[node1]) > 1:
                MultiXfmrTank[node1] = len(Xfmr_tank_name[node1])

        # add in PowerTransformerEnd transformer contribution if applicable
        if node1 in Xfmr_end_name:
            for xfmr in Xfmr_end_name[node1]:
                print('Checking end transformer name: ' + xfmr +', enum: ' + str(Enum_end[xfmr]))
                if Enum_end[xfmr][bus] == 2:
                    num_elem += 1
                    ratedU_ratio = RatedU_end[xfmr][1]/RatedU_end[xfmr][2]
                    ratedU_sq = ratedU_ratio*ratedU_ratio
                    sum_shunt_imag += -B_S[xfmr]*ratedU_sq
                    print('Adding tank transformer imag contribution: ' + str(-B_S[xfmr]*ratedU_sq))
                    sum_shunt_real += G_S[xfmr]*ratedU_sq
                    print('Adding tank transformer real contribution: ' + str(G_S[xfmr]*ratedU_sq))
            if len(Xfmr_end_name[node1]) > 1:
                MultiXfmrEnd[node1] = len(Xfmr_end_name[node1])

        compareShuntImag(sum_shunt_imag, Yshunt.imag)
        compareShuntReal(sum_shunt_real, Yshunt.real)

        if num_elem == 0:
           print('    *** No shunt elements for node: ' + node1)
           #sys.exit()
        elif num_elem > 1:
           print('    *** Multiple shunt elements for node: ' + node1)
           MultiElem[node1] = num_elem
           #sys.exit()

    print("\nSummary for ShuntElement elements:", flush=True)
    print("\nSummary for ShuntElement elements:", file=logfile)

    countImag = greenCountImag + yellowCountImag + redCountImag
    print("\nImag \u001b[32m\u25cf\u001b[37m  count: " + str(greenCountImag), flush=True)
    print("\nImag \u25cb  count: " + str(greenCountImag), file=logfile)
    print("Imag \u001b[33m\u25cf\u001b[37m  count: " + str(yellowCountImag), flush=True)
    print("Imag \u25d1  count: " + str(yellowCountImag), file=logfile)
    print("Imag \u001b[31m\u25cf\u001b[37m  count: " + str(redCountImag), flush=True)
    print("Imag \u25cf  count: " + str(redCountImag), file=logfile)

    countReal = greenCountReal + yellowCountReal + redCountReal
    print("\nReal \u001b[32m\u25cf\u001b[37m  count: " + str(greenCountReal), flush=True)
    print("\nReal \u25cb  count: " + str(greenCountReal), file=logfile)
    print("Real \u001b[33m\u25cf\u001b[37m  count: " + str(yellowCountReal), flush=True)
    print("Real \u25d1  count: " + str(yellowCountReal), file=logfile)
    print("Real \u001b[31m\u25cf\u001b[37m  count: " + str(redCountReal), flush=True)
    print("Real \u25cf  count: " + str(redCountReal), file=logfile)

    if len(MultiElem) > 0:
        print("\nMultiple elements for the following nodes: " + str(MultiElem))
    if len(MultiCap) > 0:
        print("\nMultiple capacitors for the following nodes: " + str(MultiCap))
    if len(MultiXfmrTank) > 0:
        print("\nMultiple tank transformers for the following nodes: " + str(MultiXfmrTank))
    if len(MultiXfmrEnd) > 0:
        print("Multiple end transformers for the following nodes: " + str(MultiXfmrEnd))

    print("\nFinished validation for ShuntElement elements", flush=True)
    print("\nFinished validation for ShuntElement elements", file=logfile)

    return countImag, countReal


def start(log_file, feeder_mrid, model_api_topic, simulation_id):
    global logfile
    logfile = log_file

    print("\nSHUNT_ELEMENT_VALIDATOR starting!!!---------------------------------------------")
    print("\nSHUNT_ELEMENT_VALIDATOR starting!!!---------------------------------------------", file=logfile)

    SPARQLManager = getattr(importlib.import_module('shared.sparql'), 'SPARQLManager')

    gapps = GridAPPSD()

    sparql_mgr = SPARQLManager(gapps, feeder_mrid, model_api_topic, simulation_id)

    print('Querying Ybus...', flush=True)
    ysparse,nodelist = sparql_mgr.ybus_export()
    print('Processing Ybus...', flush=True)

    idx = 1
    nodes = {}
    for obj in nodelist:
        nodes[idx] = obj.strip('\"')
        idx += 1
    #print(nodes)

    Ybus = {}
    Yexp = {}
    for obj in ysparse:
        items = obj.split(',')
        if items[0] == 'Row': # skip header line
            continue
        if nodes[int(items[0])] not in Ybus:
            Ybus[nodes[int(items[0])]] = {}
        Ybus[nodes[int(items[0])]][nodes[int(items[1])]] = complex(float(items[2]), float(items[3]))
        if nodes[int(items[1])] not in Yexp:
            Yexp[nodes[int(items[1])]] = {}
        Yexp[nodes[int(items[1])]][nodes[int(items[0])]] = complex(float(items[2]), float(items[3]))
    #print(Ybus)

    print('Ybus Processed', flush=True)
    print('Querying Vnom...', flush=True)

    vnom = sparql_mgr.vnom_export()

    print('Processing Vnom...', flush=True)

    CNV = {}
    for obj in vnom:
        items = obj.split(',')
        if items[0] == 'Bus':  # skip header line
            continue

        bus = items[0].strip('"')
        basekV = float(items[1])
        #print('bus: ' + bus + ', basekV: ' + str(basekV))

        # TODO hardwire rho for basekV<0.25 for now at least
        if basekV < 0.25:
            rho = 120.0
        else:
            rho = 1000.0*basekV/math.sqrt(3.0)

        node1 = items[2].strip()
        theta = float(items[4])*math.pi/180.0
        CNV[bus+'.'+node1] = complex(rho*math.cos(theta), rho*math.sin(theta))

        node2 = items[6].strip()
        if node2 != '0':
            theta = float(items[8])*math.pi/180.0
            CNV[bus+'.'+node2] = complex(rho*math.cos(theta), rho*math.sin(theta))

            node3 = items[10].strip()
            if node3 != '0':
                theta = float(items[12])*math.pi/180.0
                CNV[bus+'.'+node3] = complex(rho*math.cos(theta), rho*math.sin(theta))

    print('Vnom Processed', flush=True)

    global greenCountImag, yellowCountImag, redCountImag
    greenCountImag = yellowCountImag = redCountImag = 0
    global greenCountReal, yellowCountReal, redCountReal
    greenCountReal = yellowCountReal = redCountReal = 0

    countImag, countReal = validate_ShuntElement_elements(sparql_mgr, Ybus, Yexp, CNV)

    # list of lists for the tabular report
    report = []

    if countImag > 0:
        VI = float(countImag - redCountImag)/float(countImag)
        report.append(["Imaginary", countImag, "{:.4f}".format(VI), greenCountImag, yellowCountImag, redCountImag])
    else:
        report.append(["Imaginary", countImag])

    if countReal > 0:
        VI = float(countReal - redCountReal)/float(countReal)
        report.append(["Real", countReal, "{:.4f}".format(VI), greenCountReal, yellowCountReal, redCountReal])
    else:
        report.append(["Real", countReal])

    print('\n', flush=True)
    print(tabulate(report, headers=["Shunt Component", "# Nodes", "VI", diffColor(0, True), diffColor(1, True), diffColor(2, True)], tablefmt="fancy_grid"), flush=True)
    print('\n', file=logfile)
    print(tabulate(report, headers=["Shunt Component", "# Nodes", "VI", diffColor(0, False), diffColor(1, False), diffColor(2, False)], tablefmt="fancy_grid"), file=logfile)

    print('\nSHUNT_ELEMENT_VALIDATOR DONE!!!', flush=True)
    print('\nSHUNT_ELEMENT_VALIDATOR DONE!!!', file=logfile)

    return


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
    log_file = open('shunt_element_validator.log', 'w')

    start(log_file, feeder_mrid, model_api_topic, simulation_id)


if __name__ == "__main__":
    _main()
