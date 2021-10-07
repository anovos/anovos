#!/bin/bash


end=$((SECONDS+$1))

while [ $SECONDS -lt $end ]; do
	output=(`ps -ax | grep pyspark-shell | cut -f2 -d" "`)
	output_str=`ps -ax | grep pyspark-shell | cut -f2 -d" "`
        if [ ${#output[@]} -eq 1 ]
        then
            sleep 1
        else
	    final_cmd=`kill -9 $output_str`
        fi
done
