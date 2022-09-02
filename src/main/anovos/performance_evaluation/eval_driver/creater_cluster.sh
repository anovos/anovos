#!/bin/bash

machine_type="$1"
node_count="$2"
bootstrap="$3"
c_name="$4"
uuid=$(uuidgen)
output_file="cluster_details_""$node_count""_""$uuid"".txt"

aws emr create-cluster --applications Name=Hive Name=Spark Name=TensorFlow Name=Hadoop --tags 'AUTO_NAME=mw_engr_anovos' 'AUTO_TYPE=EMR' 'Name=mw_engr_anovos''MONITOR_EMAIL=krishnachur@mobilewalla.com' --ec2-attributes '{"KeyName":"mw","InstanceProfile":"emr-ec2-defaultrole","ServiceAccessSecurityGroup":"sg-0eac20a5c0142d31f","SubnetId":"subnet-00ed9fd09a9117fec","EmrManagedSlaveSecurityGroup":"sg-07d6ffde90ed91a53","EmrManagedMasterSecurityGroup":"sg-03019904b1427e7ad","AdditionalMasterSecurityGroups":["sg-0334a0e33c482cf54"]}' --release-label emr-5.33.0 --log-uri 's3n://aws-logs-361166629815-us-east-1/elasticmapreduce/' --instance-groups '[{"InstanceCount":'\"$node_count\"',"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":800,"VolumeType":"gp2"},"VolumesPerInstance":1}],"EbsOptimized":true},"InstanceGroupType":"CORE","InstanceType":'\"$machine_type\"',"Name":"On Demand Core"},{"InstanceCount":1,"EbsConfiguration":{"EbsBlockDeviceConfigs":[{"VolumeSpecification":{"SizeInGB":800,"VolumeType":"gp2"},"VolumesPerInstance":1}],"EbsOptimized":true},"InstanceGroupType":"MASTER","InstanceType":'\"$machine_type\"',"Name":"Master"}]' --bootstrap-actions '[{"Path":'\"$bootstrap\"',"Name":"python"}]' --ebs-root-volume-size 20 --service-role emr-defaultrole --enable-debugging --name "$c_name" --scale-down-behavior TERMINATE_AT_TASK_COMPLETION --region us-east-1 --profile mwdata-emr > $output_file

clusterid=`cat $output_file|/usr/bin/jq ".ClusterId" |sed 's/"//g'`
echo "$clusterid"