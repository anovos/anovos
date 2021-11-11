FROM ubuntu:18.04
ADD requirements.txt .
RUN apt-get update && apt-get install -y \
        software-properties-common
    RUN add-apt-repository ppa:deadsnakes/ppa
    RUN apt-get update && apt-get install -y \
        python3.7 \
        python3-pip
    RUN python3.7 -m pip install pip
    RUN apt-get update && apt-get install -y \
        python3-distutils \
        python3-setuptools
    RUN python3.7 -m pip install pip --upgrade pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
RUN apt-get update
RUN apt-get install -y openjdk-8-jdk
RUN apt-get update
RUN apt-get install git -y
RUN apt-get update
RUN apt-get install wget -y
RUN wget "https://downloads.apache.org/spark/spark-2.4.8/spark-2.4.8-bin-hadoop2.7.tgz"
RUN tar -xzvf spark-2.4.8-bin-hadoop2.7.tgz 
RUN rm spark-2.4.8-bin-hadoop2.7.tgz
RUN apt-get install -y python3-pip python3-dev
RUN apt-get update
RUN pip3 install --upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python
ADD bin/datapane_install ./datapane_install
RUN pip3 install "./datapane_install/datapane-0.12.0.tar.gz"
RUN python3 -m pip install -r requirements.txt
RUN cp ./datapane_install/local-report-base* /usr/local/lib/python3.7/dist-packages/datapane/resources/local_report
ADD config/log4j.properties .
ADD jars/histogrammar_2.11-1.0.20.jar .
ADD jars/histogrammar-sparksql_2.11-1.0.20.jar .
ADD dist/anovos.zip .
ADD dist/anovos.tar.gz .
ADD dist/main.py .
ADD config/configs.yaml .
ADD data/income_dataset ./data/income_dataset
ADD data/metric_dictionary.csv ./data/metric_dictionary.csv
ADD bin/spark-submit_docker.sh .
CMD ["./spark-submit_docker.sh"]
