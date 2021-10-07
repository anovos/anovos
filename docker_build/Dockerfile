FROM ubuntu:18.04
ADD requirements.txt .
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
RUN python3 -m pip install -r requirements.txt
ADD config/log4j.properties .
ADD jars/histogrammar_2.11-1.0.20.jar .
ADD jars/histogrammar-sparksql_2.11-1.0.20.jar .
ADD dist/com.zip .
ADD dist/com.tar.gz .
ADD dist/main.py .
ADD config/configs.yaml .
ADD data/income_dataset ./data/income_dataset
ADD data/data_report/stability ./output/stability
ADD bin/spark-submit_docker.sh .
ADD bin/remove_overheads_docker.sh .
CMD ["./spark-submit_docker.sh"]
#CMD ./spark-2.4.8-bin-hadoop2.7/bin/spark-submit --deploy-mode client --num-executors 1 --executor-cores 2 --executor-memory 2g --driver-memory 2G --driver-cores 1 --conf spark.driver.maxResultSize=15g --conf spark.yarn.am.memoryOverhead=1000m --conf spark.executor.memoryOverhead=2000m --conf spark.executor.extraJavaOptions=-XX:+UseCompressedOops --conf spark.executor.extraJavaOptions="-Dlog4j.configuration=file:///log4j.properties" --conf spark.driver.extraJavaOptions="-Dlog4j.configuration=file:///log4j.properties" --conf spark.kryo.referenceTracking=false --conf spark.network.timeout=18000s --conf spark.executor.heartbeatInterval=12000s --conf spark.dynamicAllocation.executorIdleTimeout=12000s --packages org.apache.spark:spark-avro_2.11:2.4.0 --conf spark.yarn.maxAppAttempts=1 --jars histogrammar_2.11-1.0.20.jar,histogrammar-sparksql_2.11-1.0.20.jar --py-files com.zip main.py configs.yaml local
