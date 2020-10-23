#!/bin/bash

#SIMREQ={\"power_system_config\":{\"Line_name\":\"_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62\"},\"service_configs\":[{\"id\":\"state-estimator\",\"user_options\":{\"use-sensors-for-estimates\":false}}]} # ieee13nodeckt using simulation

#SIMREQ={\"power_system_config\":{\"Line_name\":\"_5B816B93-7A5F-B64C-8460-47C17D6E4B0F\"},\"service_configs\":[{\"id\":\"state-estimator\",\"user_options\":{\"use-sensors-for-estimates\":false}}]} # ieee13nodecktassets using simulation

#SIMREQ={\"power_system_config\":{\"Line_name\":\"_C1C3E687-6FFD-C753-582B-632A27E28507\"},\"service_configs\":[{\"id\":\"state-estimator\",\"user_options\":{\"use-sensors-for-estimates\":false}}]} # ieee123 using simulation

#SIMREQ={\"power_system_config\":{\"Line_name\":\"_AAE94E4A-2465-6F5E-37B1-3E72183A4E44\"},\"service_configs\":[{\"id\":\"state-estimator\",\"user_options\":{\"use-sensors-for-estimates\":false}}]} # test9500new using simulation

if [ -z "$SIMREQ" ]; then
#   main.py invocation when sim_starter.py will start the simulation
    read -d "\n" SIMID SIMREQ <<< $(sim_starter/sim_starter.py $1)
else
#   main.py invocation when simulation is already started from platform viz
    SIMID=$1
fi

# only start microservices if not already running
if ! pgrep -f microservices.py -U $USER > /dev/null
then
    python3 shared/microservices.py --request "$SIMREQ" --simid "$SIMID" >/tmp/micro.out &
    # need to wait before issuing any microservices requests
    sleep 10
fi

python3 topology/topology_module.py --request "$SIMREQ" 2>&1 | tee validator.log
#./main.py "$SIMREQ" $SIMID 2>&1 | tee validator.log

# kill microservices so that it starts up with the new simulation next time
pkill -f microservices.py -U $USER

# standalone invoications of model validation modules
#python3 transformer_capacity/transformer_capacity.py --request "$SIMREQ" 2>&1 | tee validator.log
#python3 ac_line_ampacity/ac_line_ampacity.py --request "$SIMREQ" --simid $SIMID 2>&1 | tee validator.log
