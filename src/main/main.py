import copy
import subprocess
import sys
import timeit

import yaml
from com.mw.ds.data_analyzer import association_evaluator
from com.mw.ds.data_analyzer import quality_checker
from com.mw.ds.data_analyzer import stats_generator
from com.mw.ds.data_drift import drift_detector
from com.mw.ds.data_ingest import data_ingest
from com.mw.ds.data_report import report_gen_inter
from com.mw.ds.shared.spark import *


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
                    df = f(df, *value)
                else:
                    df = f(df, **value)
    return df


def save(data, write_configs, folder_name, reread=False):
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


def stats_args(all_configs, k):
    stats_configs = all_configs.get('stats_generator', None)
    write_configs = all_configs.get('write_stats', None)
    output = {}
    if (stats_configs != None) & (write_configs != None):
        read = copy.deepcopy(write_configs)
        if 'file_configs' in read:
            read['file_configs'].pop('repartition', None)
            read['file_configs'].pop('mode', None)

        if read['file_type'] == 'csv':
            read['file_configs']['inferSchema'] = True

        mainfunc_to_args = {'biasedness_detection': ['stats_mode'],
                            'IDness_detection': ['stats_unique'],
                            'outlier_detection': ['stats_unique'],
                            'correlation_matrix': ['stats_unique'],
                            'nullColumns_detection': ['stats_unique', 'stats_mode', 'stats_missing'],
                            'variable_clustering': ['stats_unique', 'stats_mode']}
        args_to_statsfunc = {'stats_unique': 'measures_of_cardinality', 'stats_mode': 'measures_of_centralTendency',
                             'stats_missing': 'measures_of_counts'}
        for arg in mainfunc_to_args.get(k, []):
            k_read = copy.deepcopy(read)
            k_read['file_path'] = k_read['file_path'] + "/data_analyzer/stats_generator/" + args_to_statsfunc[arg]
            output[arg] = k_read

    return output


def main(all_configs):
    start_main = timeit.default_timer()

    # reading main dataset
    df = ETL(all_configs.get('input_dataset'))

    write_main = all_configs.get('write_main', None)
    write_intermediate = all_configs.get('write_intermediate', None)
    write_stats = all_configs.get('write_stats', None)

    for key, args in all_configs.items():
        if (key == 'concatenate_dataset') & (args != None):
            start = timeit.default_timer()
            idfs = [df]
            for k in [e for e in args.keys() if e not in ('method')]:
                tmp = ETL(args.get(k))
                idfs.append(tmp)
            df = data_ingest.concatenate_dataset(*idfs, method_type=args.get('method'))
            df = save(df, write_intermediate, folder_name="data_ingest/concatenate_dataset", reread=True)
            end = timeit.default_timer()
            print(key, ", execution time (in secs) =", round(end - start, 4))

        if (key == 'join_dataset') & (args != None):
            start = timeit.default_timer()
            idfs = [df]
            for k in [e for e in args.keys() if e not in ('join_type', 'join_cols')]:
                tmp = ETL(args.get(k))
                idfs.append(tmp)
            df = data_ingest.join_dataset(*idfs, join_cols=args.get('join_cols'), join_type=args.get('join_type'))
            df = save(df, write_intermediate, folder_name="data_ingest/join_dataset", reread=True)
            end = timeit.default_timer()
            print(key, ", execution time (in secs) =", round(end - start, 4))

        if (key == 'stats_generator') & (args != None):
            for m in args['metric']:
                start = timeit.default_timer()
                print("\n" + m + ": \n")
                f = getattr(stats_generator, m)
                df_stats = f(df, **args['metric_args'], print_impact=False)
                save(df_stats, write_stats, folder_name="data_analyzer/stats_generator/" + m, reread=True).show(100)
                end = timeit.default_timer()
                print(key, m, ", execution time (in secs) =", round(end - start, 4))

        if (key == 'quality_checker') & (args != None):
            for subkey, value in args.items():
                if value != None:
                    start = timeit.default_timer()
                    print("\n" + subkey + ": \n")
                    f = getattr(quality_checker, subkey)
                    extra_args = stats_args(all_configs, subkey)
                    df, df_stats = f(df, **value, **extra_args, print_impact=False)
                    df = save(df, write_intermediate, folder_name="data_analyzer/quality_checker/" +
                                                                  subkey + "/dataset", reread=True)
                    save(df_stats, write_stats, folder_name="data_analyzer/quality_checker/" +
                                                            subkey + "/stats", reread=True).show(100)
                    end = timeit.default_timer()
                    print(key, subkey, ", execution time (in secs) =", round(end - start, 4))

        if (key == 'association_evaluator') & (args != None):
            for subkey, value in args.items():
                if value != None:
                    start = timeit.default_timer()
                    print("\n" + subkey + ": \n")
                    f = getattr(association_evaluator, subkey)
                    extra_args = stats_args(all_configs, subkey)
                    df_stats = f(df, **value, **extra_args, print_impact=False)
                    save(df_stats, write_stats, folder_name="data_analyzer/association_evaluator/" +
                                                            subkey + "/stats", reread=True).show(100)
                    end = timeit.default_timer()
                    print(key, subkey, ", execution time (in secs) =", round(end - start, 4))

        if (key == 'drift_detector') & (args != None):
            for subkey, value in args.items():
                if (subkey == 'drift_statistics') & (value != None):
                    start = timeit.default_timer()
                    if not value['configs']['pre_existing_source']:
                        source = ETL(value.get('source_dataset'))
                    else:
                        source = None
                    df_stats = drift_detector.drift_statistics(df, source, **value['configs'], print_impact=False)
                    save(df_stats, write_stats, folder_name="drift_detector/drift_statistics", reread=True).show(100)
                    end = timeit.default_timer()
                    print(key, subkey, ", execution time (in secs) =", round(end - start, 4))

                if (subkey == 'stabilityIndex_computation') & (value != None):
                    start = timeit.default_timer()
                    idfs = []
                    for k in [e for e in value.keys() if e not in ('configs')]:
                        tmp = ETL(value.get(k))
                        idfs.append(tmp)
                    df_stats = drift_detector.stabilityIndex_computation(*idfs, **value['configs'], print_impact=False)
                    save(df_stats, write_stats, folder_name="drift_detector/stability_index", reread=True).show(100)
                    end = timeit.default_timer()
                    print(key, subkey, ", execution time (in secs) =", round(end - start, 4))

            print("execution time w/o report (in sec) =", round(end - start, 4))

        if (key == 'report_gen_inter') & (args != None):
            print("report generation module started")
            drop_cols_viz = None
            for subkey, value in args.items():
                if subkey == 'drop_cols_viz':
                    drop_cols_viz = value
                else:
                    if value != None:
                        start = timeit.default_timer()
                        f = getattr(report_gen_inter, subkey)
                        if subkey == 'data_drift':
                            f(report_gen_inter.processed_df(df, drop_cols_viz), **value, drop_cols_viz=drop_cols_viz)
                        else:
                            f(report_gen_inter.processed_df(df, drop_cols_viz), **value)
                        end = timeit.default_timer()
                        print(key, subkey, ", execution time (in secs) =", round(end - start, 4))

    save(df, write_main, folder_name="final_dataset", reread=False)


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
