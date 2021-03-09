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


def diffColorIdx(absDiff, perDiff):
    global greenCount, yellowCount, redCount

    if absDiff<1e-3 and perDiff<0.01:
        greenCount += 1
        return 0
    elif absDiff>=1e-2 or perDiff>=0.1:
        redCount += 1
        return 2
    else:
        yellowCount += 1
        return 1


def diffPercent(shunt_elem_imag, shunt_adm_imag):
    global minPercentDiff, maxPercentDiff

    if shunt_elem_imag == 0.0:
        return 0.0

    ratio = shunt_adm_imag/shunt_elem_imag

    if ratio > 1.0:
        percent = 100.0*(ratio - 1.0)
    else:
        percent = 100.0*(1.0 - ratio)

    minPercentDiff = min(minPercentDiff, percent)
    maxPercentDiff = max(maxPercentDiff, percent)

    return percent


def compareCap(cap_name, b_per_section, shunt_adm_imag):
    print("    for capacitor: " + cap_name, flush=True)
    print("    for capacitor: " + cap_name, file=logfile)

    absDiff = abs(b_per_section - shunt_adm_imag)
    perDiff = diffPercent(b_per_section, shunt_adm_imag)
    colorIdx = diffColorIdx(absDiff, perDiff)
    print("        b_per_section:" + "{:10.6f}".format(b_per_section) + ", computed shunt admittance:" + "{:10.6f}".format(shunt_adm_imag) + "  " + diffColor(colorIdx, True), flush=True)
    print("        b_per_section:" + "{:10.6f}".format(b_per_section) + ", computed shunt admittance:" + "{:10.6f}".format(shunt_adm_imag) + "  " + diffColor(colorIdx, False), file=logfile)

    return colorIdx


def compareTrans(trans_name, shunt_elem_imag, shunt_adm_imag):
    print("    for transformer: " + trans_name, flush=True)
    print("    for transformer: " + trans_name, file=logfile)

    absDiff = abs(shunt_elem_imag - shunt_adm_imag)
    perDiff = diffPercent(shunt_elem_imag, shunt_adm_imag)
    colorIdx = diffColorIdx(absDiff, perDiff)
    print("        shunt element imag:" + "{:10.6f}".format(shunt_elem_imag) + ", computed shunt admittance:" + "{:10.6f}".format(shunt_adm_imag) + "  " + diffColor(colorIdx, True), flush=True)
    print("        shunt element imag:" + "{:10.6f}".format(shunt_elem_imag) + ", computed shunt admittance:" + "{:10.6f}".format(shunt_adm_imag) + "  " + diffColor(colorIdx, False), file=logfile)

    return colorIdx


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

    # CAPACITORS

    bindings = sparql_mgr.ShuntElement_cap_names()
    #print('SHUNT_ELEMENT_VALIDATOR ShuntElement cap_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR ShuntElement cap_names query results:', file=logfile)
    #print(bindings, file=logfile)

    # map capacitor query phase values to nodelist indexes
    ybusPhaseIdx = {'A': '.1', 'B': '.2', 'C': '.3', 's1': '.1', 's2': '.2'}

    Cap_name = {}
    B_per_section = {}
    for obj in bindings:
        cap_name = obj['cap_name']['value']
        b_per_section = float(obj['b_per_section']['value'])
        bus = obj['bus']['value']
        phase = 'ABC' # no phase specified indicates 3-phase
        if 'phase' in obj:
            phase = obj['phase']['value']
        mode = 'voltage'
        if 'mode' in obj:
            mode = obj['mode']['value']
        #print('cap_name: ' + cap_name + ', b_per_section: ' + str(b_per_section) + ', phase: ' + phase + ', mode: ' + mode)

        if mode != 'timeScheduled':
            if phase == 'ABC': # 3-phase
                Cap_name[bus+'.1'] = cap_name
                Cap_name[bus+'.2'] = cap_name
                Cap_name[bus+'.3'] = cap_name
                B_per_section[bus+'.1'] = b_per_section
                B_per_section[bus+'.2'] = b_per_section
                B_per_section[bus+'.3'] = b_per_section
            else: # specified phase only
                Cap_name[bus+ybusPhaseIdx[phase]] = cap_name
                B_per_section[bus+ybusPhaseIdx[phase]] = b_per_section

    # TRANSFORMERS

    bindings = sparql_mgr.TransformerTank_xfmr_rated()
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_rated query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_rated query results:', file=logfile)
    #print(bindings, file=logfile)

    # TransformerTank queries
    RatedS = {}
    RatedU = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        enum = int(obj['enum']['value'])
        if xfmr_name not in RatedS:
            RatedS[xfmr_name] = {}
            RatedU[xfmr_name] = {}

        RatedS[xfmr_name][enum] = int(obj['ratedS']['value'])
        RatedU[xfmr_name][enum] = int(obj['ratedU']['value'])
        print('xfmr_name: ' + xfmr_name + ', enum: ' + str(enum) + ', ratedS: ' + str(RatedS[xfmr_name][enum]) + ', ratedU: ' + str(RatedU[xfmr_name][enum]))

    bindings = sparql_mgr.TransformerTank_xfmr_nlt()
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_nlt query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_nlt query results:', file=logfile)
    #print(bindings, file=logfile)

    I_exciting = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        I_exciting[xfmr_name] = float(obj['i_exciting']['value'])
        print('xfmr_name: ' + xfmr_name + ', i_exciting: ' + str(I_exciting[xfmr_name]))

    bindings = sparql_mgr.TransformerTank_xfmr_names()
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR TransformerTank xfmr_names query results:', file=logfile)
    #print(bindings, file=logfile)

    Xfmr_tank_name = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        bus = obj['bus']['value'].upper()
        phase = obj['phase']['value']
        if phase == 'ABC':
            Xfmr_tank_name[bus+'.1'] = xfmr_name
            Xfmr_tank_name[bus+'.2'] = xfmr_name
            Xfmr_tank_name[bus+'.3'] = xfmr_name
        else:
            Xfmr_tank_name[bus+ybusPhaseIdx[phase]] = xfmr_name

        print('xfmr_tank_name: ' + xfmr_name + ', bus: ' + bus + ', phase: ' + phase)

    # TransformerEnd queries
    bindings = sparql_mgr.PowerTransformerEnd_xfmr_admittances()
    #print('SHUNT_ELEMENT_VALIDATOR PowerTransformerEnd xfmr_admittances query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR PowerTransformerEnd xfmr_admittances query results:', file=logfile)
    #print(bindings, file=logfile)

    B_S = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        B_S[xfmr_name] = float(obj['b_S']['value'])

        print('xfmr_name: ' + xfmr_name + ', b_S: ' + str(B_S[xfmr_name]))

    bindings = sparql_mgr.PowerTransformerEnd_xfmr_names()
    #print('SHUNT_ELEMENT_VALIDATOR PowerTransformerEnd xfmr_names query results:', flush=True)
    #print(bindings, flush=True)
    #print('SHUNT_ELEMENT_VALIDATOR PowerTransformerEnd xfmr_names query results:', file=logfile)
    #print(bindings, file=logfile)

    Xfmr_end_name = {}
    for obj in bindings:
        xfmr_name = obj['xfmr_name']['value']
        bus = obj['bus']['value'].upper()
        Xfmr_end_name[bus+'.1'] = xfmr_name
        Xfmr_end_name[bus+'.2'] = xfmr_name
        Xfmr_end_name[bus+'.3'] = xfmr_name

        print('xfmr_end_name: ' + xfmr_name + ', bus: ' + bus)

    # Final validation -- check all nodes for shunt elements

    global minPercentDiff, maxPercentDiff
    minPercentDiff = sys.float_info.max
    maxPercentDiff = -sys.float_info.max
    global greenCount, yellowCount, redCount
    greenCount = yellowCount = redCount = 0

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

        shunt_adm = numsum/CNV[node1]

        #print('cnv(' + node1 + ') = ' + str(CNV[node1]))
        #print('shunt_admittance(' + node1 + ') = ' + str(shunt_adm))

        # criteria to recognize shunt elements based on admittance
        if abs(shunt_adm.real)>1.0e-3 or abs(shunt_adm.imag)>1.0e-3:
            print('\nValidating shunt element node: ' + node1, flush=True)
            print('\nValidating shunt element node: ' + node1, file=logfile)

            if node1 in B_per_section:
                # validate capacitor shunt element
                #print('*** Found capacitor shunt element, comparing b_per_section: ' + str(B_per_section[node1]) + ', with shunt admittance: ' + str(shunt_adm.imag))
                compareCap(Cap_name[node1], B_per_section[node1], shunt_adm.imag)

            elif node1 in Xfmr_tank_name:
                # validate TransformerTank transformer
                xfmr_name = Xfmr_tank_name[node1]
                zBaseS = (RatedU[xfmr_name][2]*RatedU[xfmr_name][2])/RatedS[xfmr_name][2]
                shunt_elem_imag = -I_exciting[xfmr_name]/(100.0*zBaseS)
                compareTrans(xfmr_name, shunt_elem_imag, shunt_adm.imag)

            elif node1 in Xfmr_end_name:
                # validate PowerTransformerEnd transformer
                xfmr_name = Xfmr_end_name[node1]
                shunt_elem_imag = B_S[xfmr_name]
                compareTrans(xfmr_name, shunt_elem_imag, shunt_adm.imag)

    return


    # list of lists for the tabular report
    report = []

    #PowerTransformerEnd_xfmrs = 0
    PowerTransformerEnd_xfmrs = validate_PowerTransformerEnd_xfmrs(sparql_mgr, Ybus)
    if PowerTransformerEnd_xfmrs > 0:
        count = greenCount + yellowCount + redCount
        VI = float(count - redCount)/float(count)
        report.append(["PowerTransformerEnd", PowerTransformerEnd_xfmrs, "{:.4f}".format(VI), greenCount, yellowCount, redCount])
    else:
        report.append(["PowerTransformerEnd", PowerTransformerEnd_xfmrs])

    print('\n', flush=True)
    print(tabulate(report, headers=["Transformer Type", "# Transformers", "VI", diffColor(0, True), diffColor(1, True), diffColor(2, True)], tablefmt="fancy_grid"), flush=True)
    print('\n', file=logfile)
    print(tabulate(report, headers=["Transformer Type", "# Transformers", "VI", diffColor(0, False), diffColor(1, False), diffColor(2, False)], tablefmt="fancy_grid"), file=logfile)

    print('\nSHUNT_ELEMENT_VALIDATOR DONE!!!', flush=True)
    print('\nSHUNT_ELEMENT_VALIDATOR DONE!!!', file=logfile)


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
