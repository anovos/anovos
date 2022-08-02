#!/usr/bin/env bash

sudo docker build . -t anovos-demo

sudo docker run --name anovos_demo -t -i -v $(PWD):/temp anovos-demo:latest

sudo docker cp anovos_demo:/report_stats/ml_anovos_report.html .
open ml_anovos_report.html
