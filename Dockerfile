FROM ubuntu:18.04

RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y openjdk-8-jdk git wget python3-pip python3-dev python3.7 python3-distutils python3-setuptools

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

RUN pip install -r requirements.txt && pip install -r dev_requirements.txt
