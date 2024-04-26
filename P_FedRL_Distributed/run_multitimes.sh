#!/bin/bash

for j in $(seq 0 4); do
	echo "Starting server $j"
	python Server.py $j &
	sleep 3
	
	for i in $(seq 0 9); do
		echo "Starting client $i"
		python Client.py --RLAgent-id $i &
	done
	# wait until this round went well
	wait
done
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
