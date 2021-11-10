#!/bin/bash

echo
echo ... Building Docker image ...
echo
docker build -t mw_ds_feature_machine:0.1 .
echo

echo ... Docker image successfully built ! ...
echo 
docker image ls
echo 

echo ... Running ANOVOS on Docker ...
echo 
docker run -t -i -v $(PWD):/temp mw_ds_feature_machine:0.1
echo 

echo ... Finished Running ANOVOS ! ...
echo 
echo ... Opening ANOVOS Report ...
echo 
docker_container=`docker ps -a | grep "mw_ds_feature_machine:0.1" | sed 's/ /:/g' | cut -d : -f 1`
docker cp ${docker_container}:/report_stats/ml_anovos_report.html .
open ml_anovos_report.html
