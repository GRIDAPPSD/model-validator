#!/bin/bash

# requires at least a reference to the type of simulation to use
if [ "$#" -eq 0 ]; then
    echo "Usage: ./run-topo.sh #nodes"
    echo
    exit
fi

# invocation when sim_starter.py will start the simulation
read -d "\n" SIMID SIMREQ <<< $(sim_starter/sim_starter.py $1)
# need to wait after starting a simulation so that it's initialized to
# the point it will respond to queries/subscriptions
if [[ "$1" == "9500" ]]; then
    echo "Sleeping 40 seconds to allow simulation to initialize..."
    sleep 40
else
    echo "Sleeping 10 seconds to allow simulation to initialize..."
    sleep 10
fi

# for development/testing, best to kill microservices to insure we are using
# a fresh instance for each MV invocation
# comment out this pkill when we want to use the same microservices instance
# for multple MV invocations
pkill -f microservices.py -U $USER

# only start microservices if not already running
if ! pgrep -f microservices.py -U $USER > /dev/null; then
    python3 shared/microservices.py --request "$SIMREQ" --simid "$SIMID" &
    if [[ "$1" == "9500" ]]; then
        echo "Sleeping 40 seconds to allow microservices to initialize..."
        sleep 40
    else
        echo "Sleeping 10 seconds to allow microservices to initialize..."
        sleep 10
    fi
fi

python3 topology_validator/topology_validator.py --request "$SIMREQ" 2>&1 | tee validator.log

# kill microservices so that it starts up with the new simulation next time
pkill -f microservices.py -U $USER

