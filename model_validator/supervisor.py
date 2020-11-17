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
import importlib

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# gridappsd-python module
from gridappsd import GridAPPSD
from gridappsd.simulation import Simulation
from gridappsd.topics import simulation_output_topic, simulation_log_topic

# global variables
gapps = None
appName = None
sim_id = None
feeder_mrid = None


def start_mod(args):
    # retrieve the arguments from the tuple
    mod_name, op_flag, feeder_mrid, model_api_topic, sim_id = args

    # import the module given by the configuration file
    mod_import = importlib.import_module(mod_name+'.'+mod_name)
    # find the start function in the imported module
    start_func = getattr(mod_import, 'start')

    with open(mod_name+'.log', 'w') as log_file:
        print('MV_SUPERVISOR about to call start function for: ' + mod_name, flush=True)
        log_file.write('MV_SUPERVISOR starting module: ' + mod_name + ' at: ' + str(datetime.now()) + '\n')

        try:
            if op_flag:
                start_func(log_file, feeder_mrid, model_api_topic, sim_id)
            else:
                start_func(log_file, feeder_mrid, model_api_topic)

            print('MV_SUPERVISOR finished module: ' + mod_name, flush=True)
            log_file.write('MV_SUPERVISOR finished module: ' + mod_name + ' at: ' + str(datetime.now()) + '\n')

        except:
            print('MV_SUPERVISOR failed to call start function for: ' + mod_name, flush=True)
            log_file.write('MV_SUPERVISOR failed to call start function for: ' + mod_name + '\n')


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

    print('\nMV_SUPERVISOR starting!!!-------------------------------------------------------', flush=True)

    appName = sys.argv[0]

    sim_req = sys.argv[1]
    sim_id = sys.argv[2]

    # example code for processing command line arguments, not currently used
    #for arg in sys.argv:
    #    if arg.startswith('-mag'):
    #        plotMagFlag = True
    #    elif arg[0]=='-' and arg[1:].isdigit():
    #        plotNumber = int(arg[1:])

    gapps = GridAPPSD()

    sim_config = json.loads(sim_req)

    feeder_mrid = sim_config['power_system_config']['Line_name']
    print('MV_SUPERVISOR simulation feeder_mrid: ' + feeder_mrid, flush=True)
    model_api_topic = 'goss.gridappsd.process.request.data.powergridmodel'

    print('MV_SUPERVISOR done with initialization, module handoff...', flush=True)

    with open('modules-config.json') as mod_file:
        mod_json = json.load(mod_file)

        num_modules = len(mod_json['module_configs'])
        print('MV_SUPERVISOR number of configured modules: ' + str(num_modules), flush=True)

        with ThreadPoolExecutor(max_workers=num_modules) as executor:
            for mod in mod_json['module_configs']:
                mod_name = mod['module']
                op_flag = mod['operational']

                try:
                    # invoke local start_mod function within its own thread,
                    # which will then invoke the validator module start function
                    # with standardized arguments depending on whether it's a
                    # static or operational module
                    future = executor.submit(start_mod, (mod_name, op_flag, feeder_mrid, model_api_topic, sim_id))

                except:
                    print('MV_SUPERVISOR unable to start thread for module: ' + mod_name, flush=True)

    # friendly platform disconnect
    gapps.disconnect()


if __name__ == '__main__':
    _main()

