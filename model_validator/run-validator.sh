#!/bin/bash

#SIMREQ={\"power_system_config\":{\"Line_name\":\"_49AD8E07-3BF9-A4E2-CB8F-C3722F837B62\"},\"service_configs\":[{\"id\":\"state-estimator\",\"user_options\":{\"use-sensors-for-estimates\":false}}]} # ieee13nodeckt using simulation

#SIMREQ={\"power_system_config\":{\"Line_name\":\"_5B816B93-7A5F-B64C-8460-47C17D6E4B0F\"},\"service_configs\":[{\"id\":\"state-estimator\",\"user_options\":{\"use-sensors-for-estimates\":false}}]} # ieee13nodecktassets using simulation

#SIMREQ={\"power_system_config\":{\"Line_name\":\"_C1C3E687-6FFD-C753-582B-632A27E28507\"},\"service_configs\":[{\"id\":\"state-estimator\",\"user_options\":{\"use-sensors-for-estimates\":false}}]} # ieee123 using simulation

#SIMREQ={\"power_system_config\":{\"Line_name\":\"_AAE94E4A-2465-6F5E-37B1-3E72183A4E44\"},\"service_configs\":[{\"id\":\"state-estimator\",\"user_options\":{\"use-sensors-for-estimates\":false}}]} # test9500new using simulation

if [[ -z "$SIMREQ" ]]; then
    # requires at least a reference to the type of simulation to use
    if [ "$#" -eq 0 ]; then
        echo "Usage: ./run-validator.sh #nodes"
        echo
        exit
    fi

    if [ "$#" -lt 2 ]; then
        # invocation when sim_starter.py will start the simulation
        read -d "\n" SIMID SIMREQ <<< $(sim_starter/sim_starter.py $1)
        # need to wait after starting a simulation so that it's initialized to
        # the point it will respond to queries/subscriptions
        if [[ $1 -eq 9500 ]]; then
            echo "Sleeping 40 seconds to allow simulation to initialize..."
            sleep 40
        else
            echo "Sleeping 10 seconds to allow simulation to initialize..."
            sleep 10
        fi
    else
        # invocation when sim_starter.py will return $SIMREQ and not start a simulation
        read -d "\n" SIMREQ <<< $(sim_starter/sim_starter.py $1 $2)
    fi
else
#   invocation when simulation is already started from platform viz
    SIMID=$1
fi

# if $SIMID isn't set, don't start microservices either
if [[ ! -z "$SIMID" ]]; then
    # for development/testing, best to kill microservices to insure we are using
    # a fresh instance for each MV invocation
    # comment out this pkill when we want to use the same microservices instance
    # for multple MV invocations
    pkill -f microservices.py -U $USER

    # only start microservices if not already running
    if ! pgrep -f microservices.py -U $USER > /dev/null; then
        python3 shared/microservices.py --request "$SIMREQ" --simid "$SIMID" &
        if [[ $1 -eq 9500 ]]; then
            echo "Sleeping 40 seconds to allow microservices to initialize..."
            sleep 40
        else
            echo "Sleeping 10 seconds to allow microservices to initialize..."
            sleep 10
        fi
    fi

    ./supervisor.py "$SIMREQ" $SIMID 2>&1 | tee validator.log
else
#    sleep 0  # uncomment this if commenting out the line below to avoid syntax error
    ./supervisor.py "$SIMREQ" 2>&1 | tee validator.log
fi


# standalone invocations of model validation modules--comment out supervisor.py
# call above if invoking any of these
#python3 transformer_capacity/transformer_capacity.py --request "$SIMREQ" 2>&1 | tee validator.log
#python3 ac_line_ampacity/ac_line_ampacity.py --request "$SIMREQ" --simid $SIMID 2>&1 | tee validator.log
#python3 topology_validator/topology_validator.py --request "$SIMREQ" 2>&1 | tee validator.log
#python3 line_model_validator/line_model_validator.py --request "$SIMREQ" 2>&1 | tee validator.log
#python3 power_transformer_validator/power_transformer_validator.py --request "$SIMREQ" 2>&1 | tee validator.log
#python3 switching_equipment_validator/switching_equipment_validator.py --request "$SIMREQ" 2>&1 | tee validator.log
#python3 shunt_element_validator/shunt_element_validator.py --request "$SIMREQ" --simid $SIMID 2>&1 | tee validator.log

if [[ ! -z "$SIMID" ]]; then
    # kill microservices so that it starts up with the new simulation next time
    pkill -f microservices.py -U $USER
fi
