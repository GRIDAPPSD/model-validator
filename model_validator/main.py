#!/usr/bin/env python3

# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------
"""
Created on Sept 3, 2020

@author: Gary D. Black
"""

__version__ = '0.1.0'

import sys
import time
import json
import math
import pprint

# gridappsd-python module
from gridappsd import GridAPPSD
from gridappsd.simulation import Simulation
from gridappsd.topics import simulation_output_topic, simulation_log_topic

from transformer_capacity import transformer_capacity
from ac_line_ampacity import ac_line_ampacity

# global variables
gapps = None
appName = None
sim_id = None
feeder_mrid = None
lastStatus = None
exitFlag = False


def simOutputCallback(header, message):
    msgdict = message['message']
    ts = msgdict['timestamp']
    print('MV main simulation output timestamp: ' + str(ts), flush=True)


def simLogCallback(header, message):
    global lastStatus, exitFlag

    status = message['processStatus']
    if status != lastStatus:
        lastStatus = status
        print('MV main simulation status change: ' + str(status), flush=True)
        if status=='COMPLETE' or status=='CLOSED':
            print('MV main simulation done, exiting', flush=True)
            exitFlag = True


def _main():
    global appName, sim_id, feeder_mrid, gapps

    if len(sys.argv)<2 or '-help' in sys.argv:
        usestr =  '\nUsage: ' + sys.argv[0] + ' simID simReq\n'
        usestr += '''
Optional command line arguments:
        -ex[ample]: example command line argument -help user documentation as
         a placeholder until we have real arguments
        -conf[ig]: do we want an optional config file to control what is done
         for a model validator application in addition to being able to specify
         that via command line?
        -print: command line options to control output might be useful
        -help: show this usage message
        '''
        print(usestr, flush=True)
        exit()

    appName = sys.argv[0]

    startFlag = False
    if sys.argv[1] == '--start':
        startFlag = True
        sim_config_file = './simulation_config_files/' + sys.argv[2] + '-config.json'
    else:
        sim_req = sys.argv[1]
        sim_id = sys.argv[2]

    # example code for processing command line arguments, not currently used
    plotConfigFlag = False
    plotBusFlag = False
    plotPhaseFlag = False
    plotTitleFlag = False
    plotBusList = []
    for arg in sys.argv:
        if plotBusFlag:
            plotBusList.append(arg)
            plotBusFlag = False
        elif plotPhaseFlag:
            plotPhaseList.append(arg.upper())
            plotPhaseFlag = False
        elif plotTitleFlag:
            plotTitle = arg
            plotTitleFlag = False
        elif arg == '-legend':
            plotLegendFlag = True
        elif arg.startswith('-mag'):
            plotMagFlag = True
        elif arg[0]=='-' and arg[1:].isdigit():
            plotNumber = int(arg[1:])
            plotStatsFlag = False
            plotConfigFlag = False

    gapps = GridAPPSD()

    if startFlag:
        with open(sim_config_file) as config_fp:
            sim_config = json.load(config_fp)

        print('MV main initializing simulation from: ' + sim_config_file, flush=True)
        sim = Simulation(gapps, sim_config)
        print('MV main about to start simulation...', flush=True)
        sim.start_simulation()
        sim_id = sim.simulation_id
        print('MV main simulation started with id: ' + sim_id, flush=True)
    else:
        sim_config = json.loads(sim_req)

    # example code for parsing the service_configs and user_options
    for jsc in sim_config['service_configs']:
        if jsc['id'] == 'gridappsd-sensor-simulator':
            sensorSimulatorRunningFlag = True
        elif jsc['id'] == 'state-estimator':
            useSensorsForEstimatesFlag = jsc['user_options']['use-sensors-for-estimates']

    # example code to subscribe to simulation measurements
    gapps.subscribe(simulation_output_topic(sim_id), simOutputCallback)
    gapps.subscribe(simulation_log_topic(sim_id), simLogCallback)

    feeder_mrid = sim_config['power_system_config']['Line_name']
    print('MV main simulation feeder_mrid: ' + feeder_mrid, flush=True)
    model_api_topic = 'goss.gridappsd.process.request.data.powergridmodel'

    print('MV main done with initialization, module handoff...', flush=True)

    # invoke Shiva's transformer capacity module
    transformer_capacity.start(feeder_mrid, model_api_topic)

    # invoke Shiva's AC line ampacity module
    ac_line_ampacity.start(feeder_mrid, model_api_topic, sim_id)

    # TODO need to block here to avoid hitting the disconnect and exiting
    # depending on what we want the main model-validator main app to do,
    # that could be as simple as just a while loop that calls sleep repeatedly
    # like the sample app allowing the other threads that process messages
    # to get the needed CPU time
    while not exitFlag:
        time.sleep(0.1)

    # for an app with a GUI though, it should enter the GUI event processing
    # loop at this point
    #plt.show()

    # depending on what is done to block above, this disconnect may never
    # be reached.  It will for a GUI app though so it's nice to free resources
    gapps.disconnect()


if __name__ == '__main__':
    _main()

