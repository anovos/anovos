#!/usr/bin/env bash

MY_PYTHON_VERSION="${PYTHON_VERSION:-3.7}"
MY_SPARK_VERSION="${SPARK_VERSION:-2.4.8}"
MY_SPARK_CHECKSUM="${SPARK_CHECKSUM:-752C4D4D8FE1D72F5BA01F40D22DF35698585BD17ED4749F6065B0039FF40DB7FF8EA87DC0FB5B1EC03871E427A002581EC12F486392B92B88643D4243908E55}"
MY_JDK_VERSION="${JDK_VERSION:-8}"
MY_HADOOP_VERSION="${HADOOP_VERSION:-2.7}"

# Get the official Jupyter Dockerfiles
rm -rf docker-stacks && git clone --depth 1 https://github.com/jupyter/docker-stacks.git

# Build the Jupyter base images with the specified Python version
sudo docker build ./docker-stacks/base-notebook -t base-notebook:"${MY_PYTHON_VERSION}" --build-arg PYTHON_VERSION="${MY_PYTHON_VERSION}"
sudo docker build ./docker-stacks/minimal-notebook -t minimal-notebook:"${MY_PYTHON_VERSION}" --build-arg BASE_CONTAINER=base-notebook:"${MY_PYTHON_VERSION}"
sudo docker build ./docker-stacks/scipy-notebook -t scipy-notebook:"${MY_PYTHON_VERSION}" --build-arg BASE_CONTAINER=minimal-notebook:"${MY_PYTHON_VERSION}"

# Build the Anovos notebook image
sudo docker build ./docker-stacks/pyspark-notebook -t anovos-notebook-"${MY_SPARK_VERSION}" \
  --build-arg BASE_CONTAINER=scipy-notebook:"${MY_PYTHON_VERSION}" \
  --build-arg spark_version="${MY_SPARK_VERSION}" \
  --build-arg spark_checksum="${MY_SPARK_CHECKSUM}" \
  --build-arg openjdk_version="${MY_JDK_VERSION}" \
  --build-arg hadoop_version="${MY_HADOOP_VERSION}"

# Spin up a container and check the versions
sudo docker run -d --name anovos_notebook anovos-notebook-"${MY_SPARK_VERSION}"
sudo docker exec -it anovos_notebook /bin/bash -c "python --version && spark-submit --version && java -version"
sudo docker stop anovos_notebook
