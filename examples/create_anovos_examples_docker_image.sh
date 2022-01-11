#!/usr/bin/env bash

# Get the official Jupyter Dockerfiles
rm -rf docker-stacks && git clone --depth 1 https://github.com/jupyter/docker-stacks.git

# Build the Jupyter base images with Python 3.7.x
sudo docker build ./docker-stacks/base-notebook -t base-notebook:3.7 --build-arg PYTHON_VERSION=3.7
sudo docker build ./docker-stacks/minimal-notebook -t minimal-notebook:3.7 --build-arg BASE_CONTAINER=base-notebook:3.7
sudo docker build ./docker-stacks/scipy-notebook -t scipy-notebook:3.7 --build-arg BASE_CONTAINER=minimal-notebook:3.7

# Build the pyspark-notebook image with Python 3.7.x, Spark 2.4.8, Java 8, and Hadoop 2.7 as required by Anovos (https://gitlab.com/mwengr/mw_ds_feature_machine/-/blob/master/Dockerfile#L9)
sudo docker build ./docker-stacks/pyspark-notebook -t pyspark-notebook:anovos --build-arg BASE_CONTAINER=scipy-notebook:3.7 --build-arg spark_version=2.4.8 --build-arg openjdk_version=8 --build-arg hadoop_version=2.7 --build-arg spark_checksum=752C4D4D8FE1D72F5BA01F40D22DF35698585BD17ED4749F6065B0039FF40DB7FF8EA87DC0FB5B1EC03871E427A002581EC12F486392B92B88643D4243908E55


# Build the examples image
sudo docker build . -t anovos-examples

# Spin up a container and check the versions
sudo docker run -d --name anovos_examples anovos-examples:latest
sudo docker exec -it anovos_examples /bin/bash -c "python --version && spark-submit --version && java -version"
sudo docker stop anovos_examples
