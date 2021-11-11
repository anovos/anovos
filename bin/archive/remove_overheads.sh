#!/bin/bash


end=$((SECONDS+$1))

while [ $SECONDS -lt $end ]; do
	if [ $2 -eq 2 ] 
	then
		output=(`ps -ax | grep "[r]emove_overheads" | sed 's/^ *//g' | cut -f1 -d" "`)
		if [ ${#output[@]} -eq 1 ]
		then
	   		sleep 1
		else
			sleep 1
			for val in $output; do
				kill -15 $val
			done
		fi
	else
		output=(`ps -ax | grep "[p]yspark-shell" | sed 's/^ *//g' | cut -f1 -d" "`)
	        if [ ${#output[@]} -eq 1 ]
	        then
	        	sleep 10
	        else
	        	sleep 10
			for val in $output; do
				kill -9 $val
			done
	        fi
	fi
done
