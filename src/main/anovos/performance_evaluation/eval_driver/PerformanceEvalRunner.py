import sys
import os
import subprocess

def ends_with(string, end_str="/"):
    string = str(string)
    if string.endswith(end_str):
        return string
    return string + end_str

def launchFunctionEvaluation(all_configs):
    '''
    Launch EMR instances from remote server with specifications drawn from the evaluation_config_file.
    '''
    dataset_name = all_configs.get("dataset_name").replace(" ","_")
    idf_path = all_configs.get("dataset_path")
    ncols = all_configs.get("dataframe_size_list")
    output_parent_path = all_configs.get("output_parent_path")
    column_ratio = all_configs.get("column_type_ratio")
    functions = all_configs.get("functions")
    nodes = all_configs.get("node_counts")
    machine_type = all_configs.get("machine_type")
    bootstrap_file= all_configs.get("bootstrap_file")
    c_base_name=all_configs.get("cluster_base_name")
    zipLocation=all_configs.get("anovos_build_s3_location")
    anovosConfigFile=all_configs.get("anovos_function_config")
    mainFileLoc=all_configs.get("main_python_file_s3_location")
    run_type=all_configs.get("run_type")
    scripts_s3_loc=all_configs.get("scripts_s3_loc")

    if not os.path.exists('scripts'):
        os.makedirs('scripts')

    bash_cmd = "aws s3 cp " + ends_with(scripts_s3_loc) + " /scripts/ --recursive"
    _ = subprocess.check_output(["bash", "-c", bash_cmd])
    create_cluster_script = "./scripts/create_cluster.sh"
    function_evaluation_step_script = "./scripts/add_steps.sh"

    for node in nodes:
        #launch emr cluster with the mentioned node count
        c_name=f"{c_base_name}_{dataset_name}_for_{node}_nodes"
        for f_name in functions:
            #add each function performance evaluation as a step in the cluster that is launched in the outerloop
            stepName=f"anovos_{f_name}_evaluation"

    script_path = "/Users/krishnachur/Documents/dev/anovos_dev/anovos/src/main/anovos/performance_evaluation/eval_driver/creater_cluster.sh"

    
