#!/usr/bin/env bash

# Build the examples image
sudo docker build . -t anovos-examples

# Spin up a container and check the versions
sudo docker run -d --name anovos_examples anovos-examples:latest
sudo docker exec -it anovos_examples /bin/bash -c "python --version && spark-submit --version && java -version"
sudo docker stop anovos_examples
