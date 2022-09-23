#!/bin/bash

spark_version="$(spark/bin/spark-shell <<< sc.version | grep res0 | cut -f4 -d" ")"
echo "Spark version is ${spark_version}"
spark3_version="3.0.0"

function version_ge() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" == "$1"; }

if version_ge "${spark_version}" "${spark3_version}"; then
	avro_package="org.apache.spark:spark-avro_2.12:${spark_version}"
	histogrammar_jar="histogrammar_2.12-1.0.20.jar"
	histogrammar_sql_jar="histogrammar-sparksql_2.12-1.0.20.jar"
else
	avro_package="org.apache.spark:spark-avro_2.11:${spark_version}"
	histogrammar_jar="histogrammar_2.11-1.0.20.jar"
	histogrammar_sql_jar="histogrammar-sparksql_2.11-1.0.20.jar"
fi

spark/bin/spark-submit \
--deploy-mode client \
--num-executors 1 \
--executor-cores 2 \
--executor-memory 5g \
--driver-memory 5G \
--driver-cores 1 \
--conf spark.driver.maxResultSize=15g \
--conf spark.yarn.am.memoryOverhead=1000m \
--conf spark.executor.memoryOverhead=2000m \
--conf spark.executor.extraJavaOptions=-XX:+UseCompressedOops \
--conf spark.executor.extraJavaOptions="-Dlog4j.configuration=file:///log4j.properties" \
--conf spark.driver.extraJavaOptions="-Dlog4j.configuration=file:///log4j.properties" \
--conf spark.kryo.referenceTracking=true \
--conf spark.network.timeout=18000s \
--conf spark.executor.heartbeatInterval=12000s \
--conf spark.dynamicAllocation.executorIdleTimeout=12000s \
--conf spark.port.maxRetries=200 \
--packages "${avro_package}" \
--conf spark.yarn.maxAppAttempts=1 \
--jars ${histogrammar_jar},${histogrammar_sql_jar} \
--py-files anovos.zip \
main.py \
configs.yaml \
local
