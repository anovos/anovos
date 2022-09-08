#!/usr/bin/env bash

MY_PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
MY_SPARK_VERSION="${SPARK_VERSION:-3.2.2}"
MY_SPARK_CHECKSUM="${SPARK_CHECKSUM:-A7BDCAF598E9BCF78D7CBD2B8EA08D4363C45A4B0CDA0940E168EF7D592459DF1DDE0C33143049D58B61AF15C83E2EA2A93BCF6EC63DF46B693A36C978D57182}"
MY_JDK_VERSION="${JDK_VERSION:-11}"
MY_HADOOP_VERSION="${HADOOP_VERSION:-3.2}"

# Get the official Jupyter Dockerfiles
rm -rf docker-stacks && git clone https://github.com/jupyter/docker-stacks.git && cd docker-stacks && git checkout 1533087aaf76f8eb23603ecb8e322fc0656dcfea && cd ..

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
