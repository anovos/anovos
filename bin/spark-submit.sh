#!/bin/bash

nohup ./remove_overheads.sh 10000 1 &

spark-submit \
--deploy-mode client \
--num-executors 1000 \
--executor-cores 4 \
--executor-memory 20g \
--driver-memory 20G \
--driver-cores 4 \
--conf spark.driver.maxResultSize=15g \
--conf spark.yarn.am.memoryOverhead=1000m \
--conf spark.executor.memoryOverhead=2000m \
--conf spark.executor.extraJavaOptions=-XX:+UseCompressedOops \
--conf spark.executor.extraJavaOptions="-Dlog4j.configuration=file://$SPARK_HOME/conf/log4j.properties" \
--conf spark.driver.extraJavaOptions="-Dlog4j.configuration=file://$SPARK_HOME/conf/log4j.properties" \
--conf spark.kryo.referenceTracking=false \
--conf spark.network.timeout=18000s \
--conf spark.executor.heartbeatInterval=12000s \
--conf spark.dynamicAllocation.executorIdleTimeout=12000s \
--conf spark.port.maxRetries=200 \
--packages org.apache.spark:spark-avro_2.11:2.4.0 \
--conf spark.yarn.maxAppAttempts=1 \
--jars ../jars/histogrammar_2.11-1.0.20.jar,../jars/histogrammar-sparksql_2.11-1.0.20.jar \
--py-files com.zip \
main.py \
configs.yaml \
local

nohup ./remove_overheads.sh 10000 2 &

echo "Generating anovos final report.... "
python3 com/mw/ds/data_report/report_gen_final.py
