import boto3
from datetime import datetime, timedelta
import pandas as pd
from pandas import ExcelWriter
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

writer = ExcelWriter('test1.xlsx')

instances_response = emr_client.list_instances(ClusterId=cluster_id)
master_instance_id = emr_client.list_instances(ClusterId=cluster_id, InstanceGroupTypes=['MASTER'])["Instances"][0]['Ec2InstanceId']
print("master_instance_id", master_instance_id)

start_datetime = datetime.now()
end_datetime = datetime.now()

metrics_map = {
    'cpu': ['cpu_time_active', 'cpu_time_guest', 'cpu_time_guest_nice', 'cpu_time_idle', 'cpu_time_iowait', 'cpu_time_irq', 'cpu_time_nice',
            'cpu_time_softirq', 'cpu_time_steal', 'cpu_time_system', 'cpu_time_user', 'cpu_usage_active', 'cpu_usage_guest', 'cpu_usage_guest_nice',
            'cpu_usage_idle', 'cpu_usage_iowait', 'cpu_usage_irq', 'cpu_usage_nice', 'cpu_usage_softirq', 'cpu_usage_steal', 'cpu_usage_user', 'cpu_usage_system'],
    'disk': ['disk_free', 'disk_inodes_free', 'disk_inodes_total', 'disk_inodes_used', 'disk_total', 'disk_used', 'disk_used_percent'],
    'diskio': ['diskio_iops_in_progress', 'diskio_io_time', 'diskio_reads', 'diskio_read_bytes', 'diskio_read_time', 'diskio_writes', 'diskio_write_bytes', 'diskio_write_time'],
    'mem': ['mem_active', 'mem_available', 'mem_available_percent', 'mem_buffered', 'mem_cached', 'mem_free', 'mem_inactive', 'mem_total', 'mem_used', 'mem_used_percent'],
    'net': ['net_bytes_recv', 'net_bytes_sent', 'net_drop_in', 'net_drop_out', 'net_err_in', 'net_err_out', 'net_packets_sent', 'net_packets_recv'],
    'netstat': ["netstat_tcp_close", "netstat_tcp_close_wait", "netstat_tcp_closing", "netstat_tcp_established", "netstat_tcp_fin_wait1", "netstat_tcp_fin_wait2",
                "netstat_tcp_last_ack", "netstat_tcp_listen", "netstat_tcp_none", "netstat_tcp_syn_sent", "netstat_tcp_syn_recv", "netstat_tcp_time_wait", "netstat_udp_socket"],
    'processes': ["processes_blocked", "processes_dead", "processes_idle", "processes_paging", "processes_running", "processes_sleeping", "processes_stopped",
                  "processes_total", "processes_total_threads", "processes_wait", "processes_zombies"],
    'swap': ["swap_free", "swap_used", "swap_used_percent"]
}
lst = [metric for k,v in metrics_map.items() for metric in v]
print("length of metric", len(lst))

# get steps in a cluster, excluding "Setup hadoop debugging"
response = emr_client.list_steps(ClusterId=cluster_id)
steps = [step for step in response['Steps'] if step["Name"] != "Setup hadoop debugging"]

for step in steps:
    timeline = step["Status"]["Timeline"]
    start_datetime = timeline["StartDateTime"]
    end_datetime = datetime.now() if not "EndDateTime" in timeline else timeline["EndDateTime"]
    end_datetime = end_datetime + timedelta(minutes=1)
    print(start_datetime, end_datetime)
    # response = emr_client.list_instances(ClusterId=cluster_id)

    # loop through each instance in the cluster
    for instance in instances_response["Instances"]:
        print("instance id", instance["Ec2InstanceId"])
        ec2_instance_id = instance["Ec2InstanceId"]
        df = pd.DataFrame()
        for key, value in metrics_map.items():
            for metric in value:
                response = cloudwatch_client.get_metric_statistics(
                    Namespace='CWAgent',
                    MetricName=metric,
                    Dimensions=[{'Name': 'InstanceId', 'Value': ec2_instance_id}],
                    StartTime=start_datetime,
                    EndTime=end_datetime,
                    Period=60,
                    Statistics=['Maximum', 'Average']
                )
                # sort based on timestamp
                arr = list(sorted(response['Datapoints'], key=lambda x: x['Timestamp']))
                if df.empty:
                    # if df is empty, then initilize the df by adding the index
                    d_range = [datetime.strftime(item['Timestamp'], '%Y-%m-%d %H:%M') for item in arr]
                    df = pd.DataFrame(index=d_range) # index is timestamp

                if arr:
                    # add the average and maximum value for each metrics
                    unit = '(Unit: ' + arr[0]['Unit'] + ')'
                    df[metric + ' ' + unit + ' (avg)'] = [item['Average'] for item in arr]
                    df[metric + ' ' + unit + ' (max)'] = [item['Maximum'] for item in arr]

        print(df)
        # each sheet map to each instance
        sheet_name = ec2_instance_id + '(master)' if ec2_instance_id == master_instance_id else ec2_instance_id
        df.to_excel(writer, sheet_name=sheet_name)

writer.save()