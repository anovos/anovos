import yaml
import subprocess
import copy
import sys
from com.mw.ds.shared.spark import *
from com.mw.ds.data_ingest import data_ingest
from com.mw.ds.data_analyzer import stats_generator
from com.mw.ds.data_analyzer import quality_checker
from com.mw.ds.data_analyzer import association_evaluator
from com.mw.ds.data_drift import drift_detector
from com.mw.ds.data_report import report_preprocessing
from com.mw.ds.data_report.report_preprocessing import save_stats
from com.mw.ds.data_report.report_generation import anovos_report
import timeit

def ETL(args):
    f = getattr(data_ingest, 'read_dataset')
    read_args = args.get('read_dataset', None)
    if read_args:
        df = f(**read_args)
    else:
        raise TypeError('Invalid input for reading dataset')

    for key, value in args.items():
        if key != 'read_dataset':
            if value != None:
                f = getattr(data_ingest, key)
                if isinstance(value, list):
                    df = f(df,*value)
                else:
                    df = f(df,**value)       
    return df

def save(data,write_configs,folder_name,reread=False):

    if write_configs:

        if 'file_path' not in write_configs:
            raise TypeError('file path missing for writing data')
            
        write = copy.deepcopy(write_configs)
        write['file_path'] = write['file_path'] + "/" + folder_name
        data_ingest.write_dataset(data, **write)

        if reread:

            read = copy.deepcopy(write)
            if 'file_configs' in read:
                read['file_configs'].pop('repartition', None)
                read['file_configs'].pop('mode', None)
            data = data_ingest.read_dataset(**read)
            return data


def stats_args(all_configs,k):
    stats_configs = all_configs.get('stats_generator',None)
    write_configs = all_configs.get('write_stats',None)
    report_inputPath = ''
    report_configs = all_configs.get('report_preprocessing',None)
    if report_configs != None:
        if 'master_path' not in report_configs:
            raise TypeError('Master path missing for saving report statistics')
        else:
            report_inputPath = report_configs.get('master_path')
    
    if stats_configs:
        mainfunc_to_args = {'biasedness_detection': ['stats_mode'],
                     'IDness_detection': ['stats_unique'],
                     'outlier_detection': ['stats_unique'],
                     'correlation_matrix': ['stats_unique'],
                     'nullColumns_detection': ['stats_unique','stats_mode','stats_missing'],
                     'variable_clustering':['stats_unique','stats_mode']}
        args_to_statsfunc = {'stats_unique':'measures_of_cardinality','stats_mode': 'measures_of_centralTendency', 
                             'stats_missing':'measures_of_counts'}
        output = {}
        for arg in mainfunc_to_args.get(k,[]):
            if report_inputPath:
                output[arg]= {'file_path': (report_inputPath + "/" + args_to_statsfunc[arg] + ".csv"),
                              'file_type': 'csv', 'file_configs': {'header':True, 'inferSchema':True}}
            else:
                if write_configs:
                    read = copy.deepcopy(write_configs)
                    if 'file_configs' in read:
                        read['file_configs'].pop('repartition', None)
                        read['file_configs'].pop('mode', None)

                    if read['file_type'] == 'csv':
                        read['file_configs']['inferSchema'] = True
            
                    read['file_path'] = read['file_path'] + "/data_analyzer/stats_generator/" + args_to_statsfunc[arg]
                    output[arg]= read
          
        return output
    
def main(all_configs,local_or_emr):

    start_main = timeit.default_timer()
    
    # reading main dataset
    df = ETL(all_configs.get('input_dataset'))
    
    write_main = all_configs.get('write_main',None)
    write_intermediate = all_configs.get('write_intermediate',None)
    write_stats = all_configs.get('write_stats',None)
    
    report_inputPath = ''
    report_configs = all_configs.get('report_preprocessing',None)
    if report_configs != None:
        if 'master_path' not in report_configs:
            raise TypeError('Master path missing for saving report statistics')
        else:
            report_inputPath = report_configs.get('master_path')

    for key, args in all_configs.items():

        if (key == 'concatenate_dataset') & (args != None):
            start = timeit.default_timer()
            idfs = [df]
            for k in [e for e in args.keys() if e not in ('method')]:
                tmp = ETL(args.get(k))
                idfs.append(tmp)
            df = data_ingest.concatenate_dataset(*idfs, method_type=args.get('method'))
            df = save(df,write_intermediate,folder_name="data_ingest/concatenate_dataset",reread=True)
            end = timeit.default_timer()
            print(key, ", execution time (in secs) =",round(end-start,4))

        if (key == 'join_dataset') & (args != None):
            start = timeit.default_timer()
            idfs = [df]
            for k in [e for e in args.keys() if e not in ('join_type', 'join_cols')]:
                tmp = ETL(args.get(k))
                idfs.append(tmp)
            df = data_ingest.join_dataset(*idfs, join_cols=args.get('join_cols'), join_type=args.get('join_type'))
            df = save(df,write_intermediate,folder_name="data_ingest/join_dataset",reread=True)
            end = timeit.default_timer()
            print(key, ", execution time (in secs) =",round(end-start,4))

        if (key == 'stats_generator') & (args != None):
            for m in args['metric']:
                start = timeit.default_timer()
                print("\n" + m + ": \n")
                f = getattr(stats_generator, m)
                df_stats = f(df,**args['metric_args'], print_impact=False)
                if report_inputPath:
                    save_stats(df_stats,report_inputPath,m, reread=True,run_type = local_or_emr).show(100)
                else:
                    save(df_stats,write_stats,folder_name="data_analyzer/stats_generator/" + m, reread=True).show(100)
                
                end = timeit.default_timer()
                print(key, m, ", execution time (in secs) =",round(end-start,4))
        
        if (key == 'quality_checker') & (args != None):
            for subkey, value in args.items():
                if value != None:
                    start = timeit.default_timer()
                    print("\n" + subkey + ": \n")
                    f = getattr(quality_checker, subkey)
                    extra_args = stats_args(all_configs,subkey)
                    df,df_stats = f(df,**value, **extra_args, print_impact=False)
                    df = save(df,write_intermediate,folder_name="data_analyzer/quality_checker/" + 
                                              subkey +"/dataset",reread=True)
                    if report_inputPath:
                        save_stats(df_stats,report_inputPath,subkey,reread=True,run_type=local_or_emr).show(100)
                    else:
                        save(df_stats,write_stats,folder_name="data_analyzer/quality_checker/" + 
                                              subkey,reread=True).show(100)
                    end = timeit.default_timer()
                    print(key, subkey, ", execution time (in secs) =",round(end-start,4))
        
       
        if (key == 'association_evaluator') & (args != None):
            for subkey, value in args.items():
                if value != None:
                    start = timeit.default_timer()
                    print("\n" + subkey + ": \n")
                    f = getattr(association_evaluator, subkey)
                    extra_args = stats_args(all_configs,subkey)
                    df_stats = f(df,**value, **extra_args, print_impact=False)
                    if report_inputPath:
                        save_stats(df_stats,report_inputPath,subkey,reread=True,run_type=local_or_emr).show(100)
                    else:
                        save(df_stats,write_stats,folder_name="data_analyzer/association_evaluator/" + 
                                                     subkey,reread=True).show(100)
                    end = timeit.default_timer()
                    print(key, subkey, ", execution time (in secs) =",round(end-start,4))

        if (key == 'drift_detector') & (args != None):
            for subkey, value in args.items():
                if (subkey == 'drift_statistics') & (value != None):
                    start = timeit.default_timer()
                    if not value['configs']['pre_existing_source']:
                        source = ETL(value.get('source_dataset'))
                    else:
                        source = None
                    df_stats = drift_detector.drift_statistics(df,source,**value['configs'],print_impact=False)
                    if report_inputPath:
                        save_stats(df_stats,report_inputPath,subkey,reread=True,run_type=local_or_emr).show(100)
                    else:
                        save(df_stats,write_stats,folder_name="drift_detector/drift_statistics",reread=True).show(100)
                    end = timeit.default_timer()
                    print(key, subkey, ", execution time (in secs) =",round(end-start,4))
                    
                if (subkey == 'stabilityIndex_computation') & (value != None):
                    start = timeit.default_timer()
                    idfs = []
                    for k in [e for e in value.keys() if e not in ('configs')]:
                        tmp = ETL(value.get(k))
                        idfs.append(tmp)
                    df_stats = drift_detector.stabilityIndex_computation(*idfs,**value['configs'],print_impact=False)
                    if report_inputPath:
                        save_stats(df_stats,report_inputPath,subkey,reread=True,run_type=local_or_emr).show(100)
                    else:
                        save(df_stats,write_stats,folder_name="drift_detector/stability_index",reread=True).show(100)
                    end = timeit.default_timer()
                    print(key, subkey, ", execution time (in secs) =",round(end-start,4))
                    
            print("execution time w/o report (in sec) =", round(end-start_main,4))
        
        if (key == 'report_preprocessing') & (args != None):
            for subkey, value in args.items():
                if (subkey == 'charts_to_objects') & (value != None):
                    start = timeit.default_timer()
                    f = getattr(report_preprocessing, subkey)
                    f(df, **value, master_path=report_inputPath,run_type=local_or_emr)
                    end = timeit.default_timer()
                    print(key, subkey, ", execution time (in secs) =",round(end-start,4))
                    
    save(df,write_main,folder_name="final_dataset",reread=False)

if __name__ == '__main__':
    config_path = sys.argv[1]
    local_or_emr = sys.argv[2]
    
    if local_or_emr == 'local':
        config_file = open(config_path, 'r')
        
    else:
        bash_cmd = "aws s3 cp " + config_path + " config.yaml"
        output = subprocess.check_output(['bash', '-c', bash_cmd])
        config_file = open('config.yaml', 'r')
    

    all_configs = yaml.load(config_file, yaml.SafeLoader)
    main(all_configs,local_or_emr)