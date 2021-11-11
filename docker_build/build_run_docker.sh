#!/bin/bash

echo && echo ... Building Docker image ... && echo
docker build -t anovos-demo . && echo

echo ... Docker image successfully built ! ... && echo 
docker image ls && echo 

echo ... Running ANOVOS on Docker ... && echo 
docker run --name anovos_demo -t -i -v $(PWD):/temp anovos-demo:latest && echo 

echo ... Finished Running ANOVOS ! ... && echo 

echo ... Opening ANOVOS Report ... && echo 
docker cp anovos_demo:/report_stats/ml_anovos_report.html .
open ml_anovos_report.html
