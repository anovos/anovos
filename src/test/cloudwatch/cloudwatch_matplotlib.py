import boto3
from datetime import datetime
import matplotlib.pyplot as plt
import sys

aws_access_key_id = sys.argv[1]
aws_secret_access_key = sys.argv[2]
region_name = sys.argv[3]
cluster_id = sys.argv[4]

session = boto3.Session(aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        region_name=region_name)

emr_client = session.client('emr')
cloudwatch_client = session.client('cloudwatch')

start_datetime = datetime.now()
end_datetime = datetime.now()

cpu_utilization_map = {}
mem_utilization_map = {}

response = emr_client.list_steps(ClusterId=cluster_id)
steps = [step for step in response['Steps'] if step["Name"] != "Setup hadoop debugging"]
for step in steps:
    timeline = step["Status"]["Timeline"]
    start_datetime = timeline["StartDateTime"]
    end_datetime = datetime.now() if not "EndDateTime" in timeline else timeline["EndDateTime"]
    print(start_datetime, end_datetime)
    response = emr_client.list_instances(ClusterId=cluster_id)

    for instance in response["Instances"]:
        ec2_instance_id = instance["Ec2InstanceId"]

        cpu_utilization_response = cloudwatch_client.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': ec2_instance_id}],
            StartTime=start_datetime,
            EndTime=end_datetime,
            Period=60,
            Statistics=['Maximum', 'Average']
        )
        cpu_arr = list(sorted(cpu_utilization_response['Datapoints'], key=lambda x: x['Timestamp']))
        cpu_utilization_map[ec2_instance_id] = cpu_arr

        mem_utilization_response = cloudwatch_client.get_metric_statistics(
            Namespace='CWAgent',
            MetricName='mem_used_percent',
            Dimensions=[{'Name': 'InstanceId', 'Value': ec2_instance_id}],
            StartTime=start_datetime,
            EndTime=end_datetime,
            Period=60,
            Statistics=['Maximum', 'Average']
        )
        mem_arr = list(sorted(mem_utilization_response['Datapoints'], key=lambda x: x['Timestamp']))
        mem_utilization_map[ec2_instance_id] = mem_arr

print('cpu_utilization_map', cpu_utilization_map)
print('mem_utilization_map', mem_utilization_map)

# for key, cpu in cpu_utilization_map.items():
#     timestamp_lst = [item['Timestamp'] for item in cpu]
#     value_lst = [item['Average'] for item in cpu]
#     plt.plot(timestamp_lst, value_lst, label=key)

for key, cpu in mem_utilization_map.items():
    timestamp_lst = [item['Timestamp'] for item in cpu]
    value_lst = [item['Average'] for item in cpu]
    plt.plot(timestamp_lst, value_lst, label=key)

plt.legend()
plt.gcf().autofmt_xdate()
plt.show()