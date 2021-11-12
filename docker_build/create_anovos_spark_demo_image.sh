#!/bin/bash


echo && echo ... Cleaning Up Current ANOVOS Docker Images ... && echo
docker stop anovos_spark_demo && docker rm anovos_spark_demo && docker rmi anovos-spark-demo

echo && echo ... Building ANOVOS Docker image ... && echo
docker build -f Dockerfile_spark_demo -t anovos-spark-demo . && echo

echo ... ANOVOS Docker image successfully built ! ... && echo 
docker image ls && echo 

echo ... Running ANOVOS on Docker ... && echo 
docker run --name anovos_spark_demo -t -i -v $(PWD):/temp anovos-spark-demo:latest && echo 

echo ... Finished Running ANOVOS ! ... && echo 

echo ... Opening ANOVOS Report ... && echo 
docker cp anovos_spark_demo:/report_stats/ml_anovos_report.html .
open ml_anovos_report.html
