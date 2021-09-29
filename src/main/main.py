import yaml
import subprocess
import copy
import os
import sys
from com.mw.ds.shared.spark import *
from com.mw.ds.shared.utils import *
from com.mw.ds.data_ingest import data_ingest
from com.mw.ds.data_analyzer import stats_generator
from com.mw.ds.data_analyzer import quality_checker
from com.mw.ds.data_analyzer import association_evaluator
from com.mw.ds.data_drift import drift_detector
from com.mw.ds.data_report import report_gen_inter
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

def main(all_configs):
    
    # reading main dataset
    df = ETL(all_configs.get('input_dataset'))
    
    write_main = all_configs.get('write_main',None)
    write_intermediate = all_configs.get('write_intermediate',None)
    write_stats = all_configs.get('write_stats',None)

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
            print(key, end-start)

        if (key == 'join_dataset') & (args != None):
            start = timeit.default_timer()
            idfs = [df]
            for k in [e for e in args.keys() if e not in ('join_type', 'join_cols')]:
                tmp = ETL(args.get(k))
                idfs.append(tmp)
            df = data_ingest.join_dataset(*idfs, join_cols=args.get('join_cols'), join_type=args.get('join_type'))
            df = save(df,write_intermediate,folder_name="data_ingest/join_dataset",reread=True)
            end = timeit.default_timer()
            print(key, end-start)

        if (key == 'stats_generator') & (args != None):
            for m in args['metric']:
                start = timeit.default_timer()
                print("\n" + m + ": \n")
                f = getattr(stats_generator, m) 
                stats = f(df,**args['metric_args'], print_impact=False)
                save(stats,write_stats,folder_name="data_analyzer/stats_generator/" + m,reread=True).show(100)
                end = timeit.default_timer()
                print(key,m, end-start)

        if (key == 'quality_checker') & (args != None):
            for subkey, value in args.items():
                if value != None:
                    start = timeit.default_timer()
                    print("\n" + subkey + ": \n")
                    f = getattr(quality_checker, subkey)
                    df,stats = f(df,**value, print_impact=False)
                    df = save(df,write_intermediate,folder_name="data_analyzer/quality_checker/" + 
                                              subkey +"/dataset",reread=True)
                    save(stats,write_stats,folder_name="data_analyzer/quality_checker/" + 
                                              subkey +"/stats",reread=True).show(100)
                    end = timeit.default_timer()
                    print(key, subkey, end-start)

        if (key == 'association_evaluator') & (args != None):
            for subkey, value in args.items():
                if value != None:
                    start = timeit.default_timer()
                    print("\n" + subkey + ": \n")
                    f = getattr(association_evaluator, subkey)
                    stats = f(df,**value, print_impact=False)
                    save(stats,write_stats,folder_name="data_analyzer/association_evaluator/" + 
                         subkey +"/stats",reread=True).show(100)
                    end = timeit.default_timer()
                    print(key, subkey, end-start)

        if (key == 'drift_detector') & (args != None):
            start = timeit.default_timer()
            if not args['drift_statistics']['pre_existing_source']:
                source = ETL(args.get('source_dataset'))
            else:
                source = None
            stats = drift_detector.drift_statistics(df,source,**args['drift_statistics'],print_impact=False)
            save(stats,write_stats,folder_name="drift_detector/drift_statistics",reread=True).show(100)
            end = timeit.default_timer()
            print(key, end-start)

        if (key == 'report_gen_inter') & (args != None):
            for subkey, value in args.items():
                if value != None:
                    start = timeit.default_timer()
                    f = getattr(report_gen_inter, subkey)
                    if subkey == 'data_drift':
                        f(df, **value)
                    else:
                        f(report_gen_inter.processed_df(df),**value)
                    end = timeit.default_timer()
                    print(key, subkey, end-start)

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
    main(all_configs)
