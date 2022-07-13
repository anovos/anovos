
from anovos.data_ingest.data_ingest import read_dataset

def build_dataset(spark, idf_path,  ncol, column_ratio):
    '''
    Build datasets of sizes mentioned in the config with the categorical and numerical columns in the ratio as mentioned in the config
    '''
    idf = read_dataset(spark, file_path = idf_path, file_type = "csv",
                  file_configs = {"header": "True", "delimiter": "," , "inferSchema": "True"})
    
    #TODO: Enable ncol based df building
    return idf

    