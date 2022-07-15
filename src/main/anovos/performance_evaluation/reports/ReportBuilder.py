import pandas as pd
def get_report(spark, execution_time_list, dataset_name, column_ratio, machine_type, node_count):
    '''
    Build a csv report from the data_path uploaded by the AnovosOperator to the EMR
    '''
    report_item_list=[]
    
    for execution_time_data in execution_time_list:
        function_data = execution_time_data.get("function_execution_data")
        for function_execution_time_data in function_data:
            report_item_dict = {}
            report_item_dict["dataset_name"] = dataset_name
            report_item_dict["column_count_in_input_data"] = str(execution_time_data.get("ncol"))
            report_item_dict["column_type_distribution(cat/num)"] = str(column_ratio)
            report_item_dict["function_name"] = function_execution_time_data.get("function")
            report_item_dict["machine_type"] = machine_type
            report_item_dict["node_count"] = node_count
            report_item_dict["execution_time(s)"] = function_execution_time_data.get("execution_time")
            report_item_list.append(report_item_dict)

    report_df = pd.DataFrame(report_item_list)
    print(report_df)
    return report_df