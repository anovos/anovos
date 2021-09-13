import yaml
import subprocess
import copy
import os
import sys
sys.path.insert(0, 'com.zip')
from spark import * 
sc.getConf().getAll()
from com.mw.ds.data_ingest.data_ingest import *
from com.mw.ds.data_analyzer import quality_checker, association_evaluator 
from com.mw.ds.data_analyzer.stats_generator import *

def ends_with(string, end_str="/"):
    '''
    :param string: "s3:mw-bucket"
    :param end_str: "/"
    :return: "s3:mw-bucket/"
    '''
    string = str(string)
    if string.endswith(end_str):
        return string
    return string + end_str

def ETL(args):
    idf = read_dataset(**args['read_file'])
    d = {'delete_cols': delete_column, 'select_cols': select_column,
         'rename_cols': rename_column, 'recast_cols': recast_column}
    for k in d.keys():
        cols = args.get(k, None)
        if cols != None:
            if isinstance(cols, list):
                idf = d[k](idf, *cols)
            else:
                idf = d[k](idf, **cols)
    return idf

def main(all_configs):
    
    # reading main dataset
    df = ETL(all_configs.get('input_dataset'))

    # Concatenating datasets
    args = all_configs.get('concatenate_datasets', None)
    if args != None:
        idfs = [df]
        for k in [e for e in args.keys() if e not in ('method')]:
            tmp = ETL(args.get(k))
            idfs.append(tmp)
        df = concatenate_dataset(*idfs, method_type=args.get('method'))

    # join datasets
    args = all_configs.get('join_datasets', None)
    if args != None:
        idfs = [df]
        for k in [e for e in args.keys() if e not in ('join_type', 'join_cols')]:
            tmp = ETL(args.get(k))
            idfs.append(tmp)
        df = join_dataset(*idfs, join_cols=args.get('join_cols'), join_type=args.get('join_type'))

    write_main = all_configs.get('output_dataset',None)
    write_intermediate = all_configs.get('intermediate_dataset',None)

    if write_main != None:
        write = copy.deepcopy(write_main)
        write['file_path'] = write['file_path'] + "/data_ingest"
        write_dataset(df, **write)

        read = copy.deepcopy(write)
        if 'file_configs' in read:
            read['file_configs'].pop('repartition', None)
            read['file_configs'].pop('mode', None)
            df = read_dataset(**read)


    args = all_configs.get('stats_generator',None)
    if args != None:
        for f in [global_summary,measures_of_counts,measures_of_centralTendency,measures_of_cardinality,
                  measures_of_percentiles,measures_of_dispersion,measures_of_shape]:
            print("\n" + f.__name__ + ": \n")
            stats = f(df,**args, print_impact=True)

            if write_intermediate != None:
                write = copy.deepcopy(write_intermediate)
                write['file_path'] = write['file_path'] + "/data_analyzer/stats_generator/" + f.__name__
                write_dataset(stats, **write)

    args = all_configs.get('quality_checker',None)
    if args != None:
        for key, value in args.items():
            if value != None:
                print("\n" + key + ": \n")
                f = getattr(quality_checker, key)
                df,stats = f(df,**value, print_impact=True)

                if write_intermediate != None:
                    write = copy.deepcopy(write_intermediate)
                    write['file_path'] = write['file_path'] + "/data_analyzer/quality_checker/" + key +"/stats"
                    write_dataset(stats, **write)
                    if value['treatment']:
                        write = copy.deepcopy(write_intermediate)
                        write['file_path'] = write['file_path'] + "/data_analyzer/quality_checker/" + key +"/dataset"
                        write_dataset(df, **write)

                        read = copy.deepcopy(write)
                        if 'file_configs' in read:
                            read['file_configs'].pop('repartition', None)
                            read['file_configs'].pop('mode', None)
                            df = read_dataset(**read)

    args = all_configs.get('association_evaluator',None)
    if args != None:
        for key, value in args.items():
            if value != None:
                print("\n" + key + ": \n")
                f = getattr(association_evaluator, key)
                stats = f(df,**value, plot=False)
                stats.show()
                if write_intermediate != None:
                    write = copy.deepcopy(write_intermediate)
                    write['file_path'] = write['file_path'] + "/data_analyzer/association_evaluator/" + key
                    write_dataset(stats, **write)                  


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
