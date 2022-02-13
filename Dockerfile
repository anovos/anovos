FROM ubuntu:18.04

RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y openjdk-8-jdk git wget python3-pip python3-dev python3.7 python3-distutils python3-setuptools

# ADD JAVA_HOME TO THE PATH
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV PATH $PATH:$JAVA_HOME/bin

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

# INSTALL SPARK
RUN wget "https://downloads.apache.org/spark/spark-2.4.8/spark-2.4.8-bin-hadoop2.7.tgz" \
    && tar -xzvf spark-2.4.8-bin-hadoop2.7.tgz \
    && mv spark-2.4.8-bin-hadoop2.7 /opt/spark \
    && rm spark-2.4.8-bin-hadoop2.7.tgz

# ADD SPARK_HOME TO PATH
ENV SPARK_HOME /opt/spark
ENV PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
ENV  PYSPARK_PYTHON /usr/bin/python3

# INSTALL DATAPANE
COPY bin/datapane_install ./datapane_install
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && python3.7 -m pip install pip --upgrade pip \
    && pip3 install "./datapane_install/datapane-0.12.0.tar.gz" \
    && cp ./datapane_install/local-report-base* /usr/local/lib/python3.7/dist-packages/datapane/resources/local_report

# ANOVOS WORKDIR
WORKDIR /anovos

COPY requirements.txt .
COPY dev_requirements.txt .
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install -r dev_requirements.txt

COPY . .