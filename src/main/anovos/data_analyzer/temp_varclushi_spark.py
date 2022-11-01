from pyspark.ml.feature import PCA, VectorAssembler, StandardScaler
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F
from anovos.shared.spark import sc
import random

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
        scaler = StandardScaler(
            inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True
        )
        scalerModel = scaler.fit(stream_df)
        scaled_df = scalerModel.transform(stream_df)

        # scaledData.select("scaledFeatures").show(5,truncate=False) # sample centered data

        # create RowMatrix and get PCs
        pca = PCA(k=n_pcs, inputCol="scaledFeatures")
        model = pca.fit(scaled_df)
        model.setOutputCol("output")
        trans_df = model.transform(scaled_df)

        # get principal Components
        princomps = (
            trans_df.select("output")
            .withColumn("pc", vector_to_array("output"))
            .select([F.col("pc")[i] for i in range(n_pcs)])
        )  # expand Vector Rows into seperate columns

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

    @staticmethod
    def _calc_tot_var(df, *clusters):
        # e.g.: *clusters = clus1, clus2 = ["col1", "col2", ...], ["col6", "col7"] - each clus is a list of column names inside that cluster

        tot_len, tot_var, tot_prop = (0,) * 3
        # tot_len: total number of columns / features in all clusters
        # tot_var: sum of the first eigenvalues (among all clusters)
        # tot_prop: weighted average of "the variance for first PC" for each cluster.

        for clus in clusters:
            if clus == []:
                continue

            c_len = len(clus)
            c_eigvals, _, _, c_varprops = VarClusHiSpark.pca(df.select(clus))
            tot_var += c_eigvals[0]
            tot_prop = (tot_prop * tot_len + c_varprops[0] * c_len) / (tot_len + c_len)
            tot_len += c_len

        # return total variance explained and proportion for clus1 and clus2
        return tot_var, tot_prop

    @staticmethod
    def _reassign(df, clus1, clus2, feat_list=None):

        if feat_list is None:
            feat_list = clus1 + clus2

        # get the initial variance (sum of first eigenvalues for the intial clusters)
        init_var = VarClusHiSpark._calc_tot_var(df, clus1, clus2)[0]
        fin_clus1, fin_clus2 = clus1[:], clus2[:]
        check_var, max_var = (init_var,) * 2

        while True:

            for feat in feat_list:
                new_clus1, new_clus2 = fin_clus1[:], fin_clus2[:]
                # reassign to anothe rcluster
                if feat in new_clus1:
                    new_clus1.remove(feat)
                    new_clus2.append(feat)
                elif feat in new_clus2:
                    new_clus1.append(feat)
                    new_clus2.remove(feat)
                else:
                    continue

                # after reassigning, get the new "tot_var" of these clusters
                new_var = VarClusHiSpark._calc_tot_var(df, new_clus1, new_clus2)[0]
                # if new variance > initial variance, the new clusters will be final cluster, and check_var will be updated to new_var (larger value)
                if new_var > check_var:
                    check_var = new_var
                    fin_clus1, fin_clus2 = new_clus1[:], new_clus2[:]

            # update max_var to the largest variance
            if max_var == check_var:
                break
            else:
                max_var = check_var

        return fin_clus1, fin_clus2, max_var

    @staticmethod
    def _reassign_rs(df, clus1, clus2, n_rs=0):

        feat_list = clus1 + clus2
        fin_rs_clus1, fin_rs_clus2, max_rs_var = VarClusHiSpark._reassign(
            df, clus1, clus2
        )

        for _ in range(n_rs):
            random.shuffle(feat_list)
            rs_clus1, rs_clus2, rs_var = VarClusHiSpark._reassign(
                df, clus1, clus2, feat_list
            )
            if rs_var > max_rs_var:
                max_rs_var = rs_var
                fin_rs_clus1, fin_rs_clus2 = rs_clus1, rs_clus2

        return fin_rs_clus1, fin_rs_clus2, max_rs_var
