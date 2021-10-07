#!/bin/bash


end=$((SECONDS+$1))

if [ $2 -eq 2 ] 
then
	output=(`ps -ax | grep [r]emove_overheads | cut -f1 -d" "`)
	output_str=`ps -ax | grep [r]emove_overheads | cut -f1 -d" "`
	if [ ${#output[@]} -eq 1 ]
	then
   		sleep 1
	else
    		final_cmd=`kill -9 $output_str`
	fi
else
	while [ $SECONDS -lt $end ]; do
		output=(`ps -ax | grep [p]yspark-shell | cut -f1 -d" "`)
		output_str=`ps -ax | grep [p]yspark-shell | cut -f1 -d" "`
	        if [ ${#output[@]} -eq 1 ]
	        then
	            sleep 10
	        else
	            sleep 10
		    final_cmd=`kill -9 $output_str`
	        fi
	done
fi
