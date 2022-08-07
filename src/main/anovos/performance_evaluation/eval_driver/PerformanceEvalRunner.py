import sys
import os
import subprocess
from time import sleep
from sympy import false, true
import yaml
import pandas as pd
import glob
import time
import math


def ends_with(string, end_str="/"):
    string = str(string)
    if string.endswith(end_str):
        return string
    return string + end_str


def launchFunctionEvaluation(eval_config_s3_path):
    '''
    Launch EMR instances from remote server with specifications drawn from the evaluation_config_file.
    '''

    bash_cmd_2 = "aws s3 cp " + eval_config_s3_path + " eval_config.yaml"
    _ = subprocess.check_output(["bash", "-c", bash_cmd_2])
    eval_config_file = "eval_config.yaml"
    with open(eval_config_file, "r") as f1:
        all_configs = yaml.load(f1, yaml.SafeLoader)

    dataset_name = all_configs.get("dataset_name").replace(" ", "_")
    output_parent_path = all_configs.get("output_parent_path")
    functions = all_configs.get("functions")
    nodes = all_configs.get("node_counts")
    machine_type = all_configs.get("machine_type")
    bootstrap_file = all_configs.get("bootstrap_file")
    c_base_name = all_configs.get("cluster_base_name")
    zipLocation = all_configs.get("anovos_build_s3_location")
    mainFileLoc = all_configs.get("main_python_file_s3_location")
    scripts_s3_loc = all_configs.get("scripts_s3_loc")
    polling_interval = all_configs.get("cluster_polling_interval")

    if not os.path.exists('scripts'):
        os.makedirs('scripts')

    bash_cmd = "aws s3 cp " + ends_with(scripts_s3_loc) + " /scripts/ --recursive"
    _ = subprocess.check_output(["bash", "-c", bash_cmd])
    create_cluster_script = "./scripts/create_cluster.sh"
    function_evaluation_step_script = "./scripts/add_steps.sh"

    clusters_launched_list = []
    for node in nodes:
        # launch emr cluster with the mentioned node count
        c_name = f"{c_base_name}_{dataset_name}_for_{node}_nodes"
        bash_cmd = f"bash {create_cluster_script} {machine_type} {node} {bootstrap_file} {c_name}"
        cluster_id = subprocess.check_output(["bash", "-c", bash_cmd])
        clusters_launched_list.append(cluster_id)
        for f_name in functions:
            # add each function performance evaluation as a step in the cluster that is launched in the outerloop
            stepName = f"anovos_{f_name}_evaluation"
            bash_cmd = f"bash {function_evaluation_step_script} {cluster_id} {zipLocation} {stepName} {mainFileLoc} {eval_config_s3_path} {node} {f_name}"
            _ = subprocess.check_output(["bash", "-c", bash_cmd])
    print(f"Clusters launched: {clusters_launched_list}")

    time.sleep(polling_interval*math.log2(polling_interval))
    jobs_finished = false
    while not jobs_finished:
        print(f"Polling job progress....\n {len(clusters_launched_list)}")
        for cluster_id in clusters_launched_list:
            bash_cmd = f"/usr/bin/aws emr describe-cluster --cluster-id {cluster_id} --profile mwdata-emr | /usr/bin/jq \".Cluster.Status.State\" | sed 's/\"//g'"
            status = subprocess.check_output(["bash", "-c", bash_cmd])
            if(str(status).upper()) == "TERMINATED":
                clusters_launched_list.remove(cluster_id)
        if(len(clusters_launched_list) == 0):
            jobs_finished = true
        else:
            time.sleep(polling_interval)

    report_path_name = ends_with(output_parent_path) + "execution_time_reports"
    all_files = glob.glob(report_path_name + "/*.csv")
    csv_files = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        csv_files.append(df)

    report_name = f"Execution_Time_Summary_{dataset_name}.csv"
    final_report_path_name = ends_with(output_parent_path) + "final_report/"
    final_df = pd.concat(csv_files)
    final_df.to_csv(report_name, index=False)
    bash_cmd = "aws s3 cp " + report_name + " " + final_report_path_name
    _ = subprocess.check_output(["bash", "-c", bash_cmd])


launchFunctionEvaluation(eval_config_s3_path=sys.argv[1])
