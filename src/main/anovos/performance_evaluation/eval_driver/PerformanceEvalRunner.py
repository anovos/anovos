import sys
import os

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

    script_path = "/Users/krishnachur/Documents/dev/anovos_dev/anovos/src/main/anovos/performance_evaluation/eval_driver/creater_cluster.sh"

    
