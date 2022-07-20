export AWS_DEFAULT_REGION=us-east-1
zipLocation="s3://mw.test/krish/anovos/performance_eval/zip/take3/anovos.zip"
stepName="anovos_correlation_matrix"
mainFileLoc="s3://mw.test/krish/anovos/performance_eval/main/RunPerformanceEvalWrapper.py"
evalConfigFile="s3://mw.test/krish/anovos/performance_eval/configs/n1/eval_config.yaml"
anovosConfigFile="s3://mw.test/krish/anovos/performance_eval/configs/configs_ieee_fraud.yaml"
run_type="emr"
node_count="5"
f_name="correlation_matrix"
clusterid=$3

/usr/bin/aws emr add-steps --cluster-id "$clusterid" --steps Type=CUSTOM_JAR,Name="$stepName",Jar="command-runner.jar",ActionOnFailure="CONTINUE","Properties":"",Args=["spark-submit","--deploy-mode","client","--num-executors","1000","--executor-cores","4","--executor-memory","30g","--driver-memory","30G","--driver-cores","4","--conf","spark.driver.maxResultSize=25g","--conf","spark.yarn.am.memoryOverhead=1000m","--conf","spark.executor.memoryOverhead=2000m","--conf","spark.kryo.referenceTracking=false","--conf","spark.network.timeout=18000s","--conf","spark.executor.heartbeatInterval=12000s","--conf","spark.dynamicAllocation.executorIdleTimeout=12000s","--conf","spark.rpc.message.maxSize=2047","--conf","spark.yarn.maxAppAttempts=1","--conf","spark.speculation=false","--conf","spark.kryoserializer.buffer.max=1024","--conf","spark.executor.extraJavaOptions=-XX:+UseG1GC","--conf","spark.driver.extraJavaOptions=-XX:+UseG1GC","--packages","org.apache.spark:spark-avro_2.11:2.4.7","--jars","s3://mw.test/krish/anovos/general/resources/jars/histogrammar_2.11-1.0.20.jar,s3://mw.test/krish/anovos/general/resources/jars/histogrammar-sparksql_2.11-1.0.20.jar","--py-files","$zipLocation","$mainFileLoc","$evalConfigFile","$anovosConfigFile","$run_type","$node_count","$f_name"] --profile mwdata-emr