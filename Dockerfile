FROM ubuntu:18.04

RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y openjdk-8-jdk git wget python3-pip python3-dev python3.7 python3-distutils python3-setuptools

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

RUN wget "https://downloads.apache.org/spark/spark-2.4.8/spark-2.4.8-bin-hadoop2.7.tgz" \
    && tar -xzvf spark-2.4.8-bin-hadoop2.7.tgz \
    && rm spark-2.4.8-bin-hadoop2.7.tgz

COPY bin/datapane_install ./datapane_install
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && python3.7 -m pip install pip --upgrade pip \
    && pip3 install "./datapane_install/datapane-0.12.0.tar.gz" \
    && cp ./datapane_install/local-report-base* /usr/local/lib/python3.7/dist-packages/datapane/resources/local_report

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY config/log4j.properties .
COPY config/configs.yaml .
COPY jars/*.jar .
COPY dist/anovos.zip .
COPY dist/main.py .
ADD dist/anovos.tar.gz .
COPY examples/data/income_dataset ./data/income_dataset
COPY data/metric_dictionary.csv ./data/metric_dictionary.csv
COPY bin/spark-submit_docker.sh .

CMD ["./spark-submit_docker.sh"]
