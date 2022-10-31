from pyspark.ml.feature import (PCA, VectorAssembler, StandardScaler)
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from anovos.shared.spark import sc

# PySpark
class VarClusHiSpark(object):

    # Not much optimization needed
    def __init__(self, df, feat_list=None, maxeigval2=1, maxclus=None, n_rs=0):

        if feat_list is None:
            self.df = df
            self.feat_list = df.columns
        else:
            self.df = df.select(feat_list)
            self.feat_list = feat_list
        self.maxeigval2 = maxeigval2
        self.maxclus = maxclus
        self.n_rs = n_rs

    @staticmethod
    def pca(df, feat_list=None, n_pcs=2):

        if feat_list is None:
            feat_list = df.columns
        else:
            df = df.select(feat_list)

        # scale dataset
        vecAssembler = VectorAssembler(inputCols=feat_list, outputCol="features")
        stream_df = vecAssembler.transform(df)

        # Standardize
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
        scalerModel = scaler.fit(stream_df)
        scaled_df = scalerModel.transform(stream_df)

        # scaledData.select("scaledFeatures").show(5,truncate=False) # sample centered data

        # create RowMatrix and get PCs
        pca = PCA(k=n_pcs, inputCol="scaledFeatures")
        model = pca.fit(scaled_df)
        model.setOutputCol("output")
        trans_df = model.transform(scaled_df)

        # get principal Components
        princomps = trans_df.select("output").withColumn("pc", vector_to_array("output")).select(
            [F.col("pc")[i] for i in range(n_pcs)])  # expand Vector Rows into seperate columns

        # get eigenvectors
        eigvecs = model.pc.toArray()  # Numpy array first

        # get variance_explained
        varprops = model.explainedVariance.toArray()  # Numpy array

        # get eigenvalues
        rm = RowMatrix(trans_df.select("scaledFeatures").rdd.map(list))
        cov = RowMatrix(sc.parallelize(rm.computeCovariance().toArray()))
        svd_model = cov.computeSVD(3, True)
        eigvals = svd_model.s.toArray()  # Numpy array

        return eigvals, eigvecs, princomps, varprops
