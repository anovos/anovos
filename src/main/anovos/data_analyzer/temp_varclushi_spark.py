from pyspark.ml.feature import PCA, VectorAssembler, StandardScaler
from pyspark.mllib.linalg.distributed import RowMatrix, DenseMatrix
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F, Window
from anovos.shared.spark import sc
import collections
import math
from factor_analyzer import Rotator
import pandas as pd
import numpy as np
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
    def correig(df, feat_list=None, n_pcs=2):

        if feat_list is None:
            feat_list = df.columns
        else:
            df = df.select(feat_list)

        if len(feat_list) <= 1:
            corr = [len(feat_list)]
            eigvals = [len(feat_list)] + [0] * (n_pcs - 1)
            eigvecs = np.array([[len(feat_list)]])
            varprops = [sum(eigvals)]
        else:
            # change rows to vector
            vecAssembler = VectorAssembler(inputCols=feat_list, outputCol="features")
            stream_df = vecAssembler.transform(df)

            # Standardize
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaledFeatures",
                withStd=True,
                withMean=True,
            )
            scalerModel = scaler.fit(stream_df)
            scaled_df = scalerModel.transform(stream_df)

            # Create Row Matrix
            rm = RowMatrix(scaled_df.select("scaledFeatures").rdd.map(list))

            # Eigenvectors
            eigvecs = rm.computePrincipalComponents(n_pcs).toArray()  # Numpy array

            # Eigvalues
            corr = rm.computeCovariance().toArray()
            cov = RowMatrix(sc.parallelize(corr))
            # svd_model = cov.computeSVD(n_pcs)
            # eigvals = svd_model.s.toArray()  # Numpy array

            # Eigenvals & Variance_Explained
            raw_model = cov.computeSVD(rm.numCols())
            raw_eigvals = raw_model.s.toArray()
            eigvals = raw_eigvals[:n_pcs]
            varprops = eigvals / sum(raw_eigvals)  # Numpyp array

        corr_df = pd.DataFrame(corr, columns=feat_list, index=feat_list)

        return eigvals, eigvecs, corr_df, varprops

    @staticmethod
    def pca(df, feat_list=None, n_pcs=2):

        if feat_list is None:
            feat_list = df.columns
        else:
            df = df.select(feat_list)

        # change rows to vector
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
        if len(feat_list) <= 1:
            princomps = scaled_df.withColumn(
                "pc", vector_to_array("scaledFeatures")
            ).select(F.col("pc")[0])
            eigvals = [len(feat_list)] + [0] * (n_pcs - 1)
            eigvecs = np.array([[len(feat_list)]])
            varprops = [sum(eigvals)]

        else:
            pca = PCA(k=n_pcs, inputCol="scaledFeatures")
            model = pca.fit(scaled_df)
            model.setOutputCol("output")
            trans_df = model.transform(scaled_df)

            # get principal Components
            princomps = (
                trans_df.select("output")
                .withColumn("pc", vector_to_array("output"))
                .select([F.col("pc")[i] for i in range(n_pcs)])
            )  # expand Vector Rows into seperate columns. PySpark dataframe

            # get eigenvectors
            eigvecs = model.pc.toArray()  # Numpy array first

            # get variance_explained
            varprops = model.explainedVariance.toArray()  # Numpy array

            # get eigenvalues
            rm = RowMatrix(scaled_df.select("scaledFeatures").rdd.map(list))
            cov = RowMatrix(
                sc.parallelize(rm.computeCovariance().toArray())
            )  # computeCovariance().toArray() is same as Correlation.corr(df, "features", "pearson") and both creates DenseMatrix
            svd_model = cov.computeSVD(n_pcs, True)
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
            c_eigvals, _, _, c_varprops = VarClusHiSpark.correig(df.select(clus))
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

    def _varclusspu(self):

        ClusInfo = collections.namedtuple(
            "ClusInfo", ["clus", "eigval1", "eigval2", "eigvecs", "varprop"]
        )
        c_eigvals, c_eigvecs, c_corrs, c_varprops = VarClusHiSpark.correig(
            self.df.select(self.feat_list)
        )

        self.corrs = c_corrs

        clus0 = ClusInfo(
            clus=self.feat_list,
            eigval1=c_eigvals[0],
            eigval2=c_eigvals[1],
            eigvecs=c_eigvecs,
            varprop=c_varprops[0],
        )
        self.clusters = collections.OrderedDict([(0, clus0)])

        while True:

            if self.maxclus is not None and len(self.clusters) >= self.maxclus:
                break

            idx = max(self.clusters, key=lambda x: self.clusters.get(x).eigval2)
            if self.clusters[idx].eigval2 > self.maxeigval2:
                split_clus = self.clusters[idx].clus
                c_eigvals, c_eigvecs, split_corrs, _ = VarClusHiSpark.correig(
                    self.df.select(split_clus)
                )
            else:
                break

            if c_eigvals[1] > self.maxeigval2:
                clus1, clus2 = [], []
                rotator = Rotator(method="quartimax")
                r_eigvecs = rotator.fit_transform(pd.DataFrame(c_eigvecs))

                comb_sigma1 = math.sqrt(
                    np.dot(
                        np.dot(r_eigvecs[:, 0], split_corrs.values), r_eigvecs[:, 0].T
                    )
                )
                comb_sigma2 = math.sqrt(
                    np.dot(
                        np.dot(r_eigvecs[:, 1], split_corrs.values), r_eigvecs[:, 1].T
                    )
                )

                for feat in split_clus:

                    comb_cov1 = np.dot(r_eigvecs[:, 0], split_corrs[feat].values.T)
                    comb_cov2 = np.dot(r_eigvecs[:, 1], split_corrs[feat].values.T)

                    corr_pc1 = comb_cov1 / comb_sigma1
                    corr_pc2 = comb_cov2 / comb_sigma2

                    if abs(corr_pc1) > abs(corr_pc2):
                        clus1.append(feat)
                    else:
                        clus2.append(feat)

                fin_clus1, fin_clus2, _ = VarClusHiSpark._reassign_rs(
                    self.df, clus1, clus2, self.n_rs
                )
                c1_eigvals, c1_eigvecs, _, c1_varprops = VarClusHiSpark.correig(
                    self.df.select(fin_clus1)
                )
                c2_eigvals, c2_eigvecs, _, c2_varprops = VarClusHiSpark.correig(
                    self.df.select(fin_clus2)
                )

                self.clusters[idx] = ClusInfo(
                    clus=fin_clus1,
                    eigval1=c1_eigvals[0],
                    eigval2=c1_eigvals[1],
                    eigvecs=c1_eigvecs,
                    varprop=c1_varprops[0],
                )
                self.clusters[len(self.clusters)] = ClusInfo(
                    clus=fin_clus2,
                    eigval1=c2_eigvals[0],
                    eigval2=c2_eigvals[1],
                    eigvecs=c2_eigvecs,
                    varprop=c2_varprops[0],
                )
            else:
                break

        return self

    def varclus(self):

        ClusInfo = collections.namedtuple(
            "ClusInfo", ["clus", "eigval1", "eigval2", "pc1", "varprop"]
        )
        c_eigvals, _, c_princomps, c_varprops = VarClusHiSpark.pca(
            self.df.select(self.feat_list)
        )
        clus0 = ClusInfo(
            clus=self.feat_list,
            eigval1=c_eigvals[0],
            eigval2=c_eigvals[1],
            pc1=c_princomps.select("pc[0]"),
            varprop=c_varprops[0],
        )
        self.clusters = collections.OrderedDict([(0, clus0)])

        while True:
            print("---- New Iteration ----")
            print(self.clusters)

            if self.maxclus is not None and len(self.clusters) >= self.maxclus:
                break

            idx = max(self.clusters, key=lambda x: self.clusters.get(x).eigval2)
            if self.clusters[idx].eigval2 > self.maxeigval2:
                split_clus = self.clusters[idx].clus
                c_eigvals, c_eigvecs, _, _ = VarClusHiSpark.pca(
                    self.df.select(split_clus)
                )
            else:
                break

            if c_eigvals[1] > self.maxeigval2:
                clus1, clus2 = [], []
                # Rotate and multiply the standardized [split_clus] dataframe with rotated eigenvecs
                rotator = Rotator(method="quartimax")
                r_eigvecs = rotator.fit_transform(c_eigvecs)

                # convert r_eigvecs to DenseMatrix
                num_rows, num_cols = c_eigvecs.shape
                r_eigvecs_m = DenseMatrix(
                    num_rows, num_cols, r_eigvecs.flatten(order="F")
                )  # DenseMatrix

                # select scaled features, scale and transform to vector
                # left = RowMatrix(scaledData.select("scaledFeatures").rdd.map(list))
                vecAssembler = VectorAssembler(
                    inputCols=split_clus, outputCol="split_clus"
                )
                split_clus_df = vecAssembler.transform(self.df.select(split_clus))

                # Standardize
                scaler = StandardScaler(
                    inputCol="split_clus",
                    outputCol="scaled_split_clus",
                    withStd=True,
                    withMean=True,
                )
                scalerModel = scaler.fit(split_clus_df)
                split_clus_df = scalerModel.transform(split_clus_df).select(
                    "scaled_split_clus"
                )
                split_clus_m = RowMatrix(split_clus_df.rdd.map(list))  # RowMatrix

                # projected PCs to r_eigvecs
                r_pcs = split_clus_m.multiply(r_eigvecs_m)  # DenseVector

                df_r_pcs = (
                    r_pcs.rows.map(lambda x: (x,))
                    .toDF(["r_pcs_vector"])
                    .withColumn("r_pc", vector_to_array("r_pcs_vector"))
                    .select([F.col("r_pc")[i] for i in range(2)])
                )  # Convert to columns r_pc[0], r_pc[1] etc
                df_r_pcs = df_r_pcs.withColumn(
                    "row_idx",
                    F.row_number().over(
                        Window.orderBy(F.monotonically_increasing_id())
                    ),
                )
                df_join = self.df.withColumn(
                    "row_idx",
                    F.row_number().over(
                        Window.orderBy(F.monotonically_increasing_id())
                    ),
                )

                df_join = df_join.join(
                    df_r_pcs, df_join.row_idx == df_r_pcs.row_idx
                ).drop(
                    "row_idx"
                )  # join to df

                for feat in split_clus:

                    corr_pc1 = df_join.stat.corr(feat, "r_pc[0]")
                    corr_pc2 = df_join.stat.corr(feat, "r_pc[1]")

                    if abs(corr_pc1) > abs(corr_pc2):
                        clus1.append(feat)
                    else:
                        clus2.append(feat)

                fin_clus1, fin_clus2, _ = VarClusHiSpark._reassign_rs(
                    self.df, clus1, clus2, self.n_rs
                )
                c1_eigvals, _, c1_princomps, c1_varprops = VarClusHiSpark.pca(
                    self.df.select(fin_clus1)
                )
                c2_eigvals, _, c2_princomps, c2_varprops = VarClusHiSpark.pca(
                    self.df.select(fin_clus2)
                )

                self.clusters[idx] = ClusInfo(
                    clus=fin_clus1,
                    eigval1=c1_eigvals[0],
                    eigval2=c1_eigvals[1],
                    pc1=c1_princomps.select("pc[0]"),
                    varprop=c1_varprops[0],
                )
                self.clusters[len(self.clusters)] = ClusInfo(
                    clus=fin_clus2,
                    eigval1=c2_eigvals[0],
                    eigval2=c2_eigvals[1],
                    pc1=c2_princomps.select("pc[0]"),
                    varprop=c2_varprops[0],
                )
            else:
                break

        return self

    def _rsquarespu(self):

        cols = ["Cluster", "Variable", "RS_Own", "RS_NC", "RS_Ratio"]
        rs_table = pd.DataFrame(columns=cols)

        sigmas = []  # sigma is the standard deviation of each cluster's PC1
        for _, clusinfo in self.clusters.items():
            c_eigvec = clusinfo.eigvecs[:, 0]
            c_sigma = math.sqrt(
                np.dot(
                    np.dot(
                        c_eigvec, self.corrs.loc[clusinfo.clus, clusinfo.clus].values
                    ),
                    c_eigvec.T,
                )
            )  # equals to std of this cluster's PC1
            sigmas.append(c_sigma)

        n_row = 0
        for i, clus_own in self.clusters.items():
            for feat in clus_own.clus:
                row = [i, feat]
                cov_own = np.dot(
                    clus_own.eigvecs[:, 0], self.corrs.loc[feat, clus_own.clus].values.T
                )  # equals cov(df[feat], clus_own.pc1) / std(df[feat])
                # since corr(df[feat], pc1) = cov(df[feat, pc1)/(std(df[feat])*std(pc1))
                if (
                    len(clus_own.clus) == 1 and feat == clus_own.clus[0]
                ):  # single-feature cluster
                    rs_own = 1
                else:
                    rs_own = (cov_own / sigmas[i]) ** 2

                rs_others = []
                for j, clus_other in self.clusters.items():
                    if j == i:
                        continue

                    cov_other = np.dot(
                        clus_other.eigvecs[:, 0],
                        self.corrs.loc[feat, clus_other.clus].values.T,
                    )  # equals to cov(df[feat], other_clus.pc1) / std(df[feat])
                    rs = (cov_other / sigmas[j]) ** 2

                    rs_others.append(rs)

                rs_nc = max(rs_others) if len(rs_others) > 0 else 0
                row += [rs_own, rs_nc, (1 - rs_own) / (1 - rs_nc)]
                rs_table.loc[n_row] = row
                n_row += 1

        return rs_table
