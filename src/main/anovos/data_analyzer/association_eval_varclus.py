from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.mllib.linalg.distributed import RowMatrix, DenseMatrix
import math
import collections
from factor_analyzer import Rotator
import pandas as pd
import numpy as np
import random


class VarClusHiSpark(object):
    """
    This class is a scalable version of [VarClusHi] [2] library to perform variable clustering on PySpark dataframes
    with necessary optimizations, and sampling is not required.

    [2]: https://github.com/jingtt/varclushi   "VarCluShi"

    Variable Clustering groups attributes that are as correlated as possible among themselves within a cluster and as
    uncorrelated as possible with attribute in other clusters. By default, it begins with all variables in a single cluster.
    It then repeats the following steps:
    1. A cluster is chosen for splitting. This cluster has the largest eigenvalue associated with the 2nd PC
    2. Find the first 2 PCs of the chosen cluster and split into 2 clusters, then perform an orthoblique rotation
    (raw quartimax rotation on the eigenvectors)
    3. Variables are iteratively re-assigned to clusters to maximize the variance:
        (a) Nearest Component Sorting (NCS) Phase: In each iteration, the cluster components are computed.
        Each variable is assigned to the rotated component with which it has the highest squared correlation
        (b) Search Phase: Assign each variable into different cluster to see if this increase the variance explained. If
         variable is re-assigned here, the components of the clusters involved are recomputed before next variable is tested
    The procedure stops when the 2nd eigenvalue of each cluster is smaller than maxeigval2 parameter.

    """

    def __init__(self, df, feat_list=None, maxeigval2=1, maxclus=None, n_rs=0):
        """
        Parameters
        ----------
        df
            PySpark Dataframe
        feat_list
            List of features to perform variable clustering e.g., ["col1","col2"].
            If feat_list is None, all columns of the input dataframe will be used.
            If feat_list is specified, only columns inside feat_list will be used.
            To run the algorithm successfully, df[feat_list] should only contain numerical values. (Default value = None)
        maxeigval2
            Maximum value of second-largest eigenvalue e.g., 1 (Default value = 1)
        maxclus
            Maximum number of clusters e.g., 20
            If maxclus is None, there will be no restrictions on total number of clusters.
            If maxclus is specified, the algorithm will stop splitting data when number of clusters reaches maxclus. (Default value = None)
        n_rs
            Number of random shuffles e.g., 0
            This parameter controls the number of random shuffle iterations after re-assignment.
            If n_rs is 0, re-assignment of each feature will be conducted with no extra random-shuffling.
            If n_rs is not 0, extra random shuffling of the features and re-assignment will be conducted. (Default value = 0)
        """

        if feat_list is None:
            self.df = df
            self.feat_list = df.columns
        else:
            self.df = df.select(feat_list)
            self.feat_list = feat_list
        self.maxeigval2 = maxeigval2
        self.maxclus = maxclus
        self.n_rs = n_rs

        if len(self.feat_list) <= 1:
            all_corr = [len(self.feat_list)]

        else:
            vecAssembler = VectorAssembler(
                inputCols=self.feat_list, outputCol="features"
            )
            stream_df = vecAssembler.transform(df)
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaledFeatures",
                withStd=True,
                withMean=True,
            )
            scalerModel = scaler.fit(stream_df)
            scaled_df = scalerModel.transform(stream_df)
            rm = RowMatrix(scaled_df.select("scaledFeatures").rdd.map(list))
            all_corr = rm.computeCovariance().toArray()

        self.corr_df = pd.DataFrame(
            all_corr, columns=self.feat_list, index=self.feat_list
        )

    def correig(self, df, feat_list=None, n_pcs=2):
        """
        This function find the correlation matrix between feat_list, calculates the top n_pcs eigenvalues, eigenvectors,
        and gets the variance-proportion.
        Parameters
        ----------
        df
            Spark dataframe
        feat_list
            list of features columns e.g. ["col1", "col2"]
            If feat_list is None, all columns of the input dataframe will be used.
            If feat_list is specified, only columns inside feat_list will be used.
            To run the algorithm successfully, df[feat_list] should only contain numerical values. (Default value = None)
        n_pcs
            number of PCs e.g. 2
            This parameter controls the size of Principal Components. e.g. If n_pcs=2, only the first 2 eigenvalues,
            eigenvectors of the correlation matrix will be extracted. (Default value = 2)
        Returns
        -------
        (top n_pcs) eigenvalues, associated eigenvectors, correlation matrix in Pandas dataframe and variance proportions
        eigvals
            Top n_pcs eigenvalues in a Numpy array e.g. [eigval1, eigval2]
        eigvecs
            The associated eigenvectors of the top n_pcs eigenvalues in a Numpy array e.g. [eigvec1, eigvec2]
            Each eigenvector is an array of size D, where D represents the number of columns of df[feat_list]
        corr_df
            Correlation matrix of stored in Pandas dataframe
            The size of this output is DxD where D represents the number of columns of df[feat_list].
            The value at index i and column j indicates the pearson's correlation score between feat_list[i] and feat_list[j]
        varprops
            The variance proportion of the top n_pcs eigenvalues in a Numpy array e.g. [varprop1, varprop2]
            The order of this array is the same with eigvals, eigvecs, with the first value varprop1 indicating the
            variance proportion of the largest eigenvalue.
        """

        if feat_list is None:
            feat_list = df.columns

        if len(feat_list) <= 1:
            corr = [len(feat_list)]
            eigvals = [len(feat_list)] + [0] * (n_pcs - 1)
            eigvecs = np.array([[len(feat_list)]])
            varprops = [sum(eigvals)]
        else:
            corr = self.corr_df.loc[feat_list, feat_list].values
            raw_eigvals, raw_eigvecs = np.linalg.eigh(corr)
            idx = np.argsort(raw_eigvals)[::-1]
            eigvals, eigvecs = raw_eigvals[idx], raw_eigvecs[:, idx]
            eigvals, eigvecs = eigvals[:n_pcs], eigvecs[:, :n_pcs]
            varprops = eigvals / sum(raw_eigvals)

        corr_df = pd.DataFrame(corr, columns=feat_list, index=feat_list)

        return eigvals, eigvecs, corr_df, varprops

    def _calc_tot_var(self, df, *clusters):
        """
        This function calculates the total variance explained of given clusters
        Parameters
        ----------
        df
            Spark dataframe
        clusters
            list of clusters e.g. [clus1, clus2]
            Each cluster is a list of feature columns which are classified into this cluster. e.g. clus1 = ["col1", "col2"]
        Returns
        -------
        tot_var, tot_prop
            tot_var: sum of the first eigenvalues among all clusters passed in. e.g. 0.5
            tot_prop: weighted average of the "variance for 1st PC" for each cluster passed in. e.g. 0.4
        """

        tot_len, tot_var, tot_prop = (0,) * 3
        for clus in clusters:
            if clus == []:
                continue

            c_len = len(clus)
            c_eigvals, _, _, c_varprops = self.correig(df.select(clus))
            tot_var += c_eigvals[0]
            tot_prop = (tot_prop * tot_len + c_varprops[0] * c_len) / (tot_len + c_len)
            tot_len += c_len
        return tot_var, tot_prop

    def _reassign(self, df, clus1, clus2, feat_list=None):
        """
        This function performs the re-assignment of each variable into different cluster.
        If the re-assignment could increase the variance explained, the components of the clusters involved are
        recomputed before next variable is tested.
        Parameters
        ----------
        df
            Spark dataframe
        clus1
            List of feature columns in first cluster e.g. ["col1", "col2"]
        clus2
            List of feature columns in second cluster e.g. ["col3", "col4"]
        feat_list
            List of features to re-assign e.g. ["feat1", "feat2"]
            If feat_list is None, all features inside clus1 and clus2 will be re-assigned
            If feat_list is specified, it should only contain columns in clus1 and/or clus2, and only the specified columns
            will be re-assigned. (Default value = None)

        Returns
        -------
        fin_clus1, fin_clus2, max_var
            fin_clus1: final cluster 1's list of feature columns e.g. ["col1", "col2"]
            fin_clus2: final cluster 2's list of feature columns e.g. ["col3", "col4"]
            max_var: the maximum variance achieved e.g. 0.9

        """

        if feat_list is None:
            feat_list = clus1 + clus2

        init_var = self._calc_tot_var(df, clus1, clus2)[0]
        fin_clus1, fin_clus2 = clus1[:], clus2[:]
        check_var, max_var = (init_var,) * 2
        while True:

            for feat in feat_list:
                new_clus1, new_clus2 = fin_clus1[:], fin_clus2[:]
                if feat in new_clus1:
                    new_clus1.remove(feat)
                    new_clus2.append(feat)
                elif feat in new_clus2:
                    new_clus1.append(feat)
                    new_clus2.remove(feat)
                else:
                    continue
                new_var = self._calc_tot_var(df, new_clus1, new_clus2)[0]
                if new_var > check_var:
                    check_var = new_var
                    fin_clus1, fin_clus2 = new_clus1[:], new_clus2[:]

            if max_var == check_var:
                break
            else:
                max_var = check_var
        return fin_clus1, fin_clus2, max_var

    def _reassign_rs(self, df, clus1, clus2, n_rs=0):
        """
        This function should be called after _reassign, and it performs random shuffling of n_rs times to run the re-assignment
        Parameters
        ----------
        df
            Spark dataframe
        clus1
            List of feature columns in first cluster e.g. ["col1", "col2"]
        clus2
            List of feature columns in second cluster e.g. ["col3", "col4"]
        n_rs
            Number of random shuffles e.g. 2
            If n_rs is 0, random shuffling of the features after re-assignment will not be conducted.
            If n_rs is >0, random shuffling of n_rs times will be conducted to perform re-assignment. (Default value = 0)

        Returns
        -------
        fin_rs_clus1, fin_rs_clus2, max_rs_var
            fin_rs_clus1: final cluster 1's list of feature columns after random shuffling e.g. ["col1", "col2"]
            fin_rs_clus2: final cluster 2's list of feature columns after random shuffling e.g. ["col3", "col4"]
            max_rs_var: the maximum variance achieved after random shuffling e.g. 0.9
        """

        feat_list = clus1 + clus2
        fin_rs_clus1, fin_rs_clus2, max_rs_var = self._reassign(df, clus1, clus2)

        for _ in range(n_rs):
            random.shuffle(feat_list)
            rs_clus1, rs_clus2, rs_var = self._reassign(df, clus1, clus2, feat_list)
            if rs_var > max_rs_var:
                max_rs_var = rs_var
                fin_rs_clus1, fin_rs_clus2 = rs_clus1, rs_clus2

        return fin_rs_clus1, fin_rs_clus2, max_rs_var

    def _varclusspark(self, spark):
        """
        This function is the main function which performs variable clustering.
        By default, it begins with all variables in a single cluster. It then repeats the following steps:
            1. A cluster is chosen for splitting. This cluster has the largest eigenvalue associated with the 2nd PC
            2. Find the first 2 PCs of the chosen cluster and split into 2 clusters, then perform an orthoblique rotation (raw quartimax rotation on the eigenvectors)
            3. Variables are iteratively re-assigned to clusters to maximize the variance
                (a) Nearest Component Sorting (NCS) Phase: In each iteration, the cluster components are computed. Each
                variable is assigned to the rotated component with which it has the highest squared correlation
                (b) Search Phase: Assign each variable into different cluster to see if this increase the variance explained.
                If variable is re-assigned here, the components of the clusters involved are recomputed before next variable is tested
        The procedure stops when the 2nd eigenvalue of each cluster is smaller than maxeigval2 parameter.

        Parameters
        ----------
        spark
            Spark Session
        Returns
        -------

        """
        """"""

        ClusInfo = collections.namedtuple(
            "ClusInfo", ["clus", "eigval1", "eigval2", "eigvecs", "varprop"]
        )
        c_eigvals, c_eigvecs, c_corrs, c_varprops = self.correig(
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
                c_eigvals, c_eigvecs, split_corrs, _ = self.correig(
                    self.df.select(split_clus)
                )
            else:
                break

            if c_eigvals[1] > self.maxeigval2:
                clus1, clus2 = [], []
                rotator = Rotator(method="quartimax")
                r_eigvecs = rotator.fit_transform(pd.DataFrame(c_eigvecs))

                num_rows, num_cols = r_eigvecs.shape
                r_eigvecs_dm = DenseMatrix(
                    num_rows, num_cols, r_eigvecs.ravel(order="F")
                )
                r_eigvecs_rm = RowMatrix(spark.sparkContext.parallelize(r_eigvecs.T))
                dm = DenseMatrix(num_rows, num_rows, split_corrs.values.ravel())

                comb_sigmas = r_eigvecs_rm.multiply(dm).multiply(r_eigvecs_dm)
                comb_sigmas = np.sqrt(
                    np.diag(comb_sigmas.rows.map(lambda row: np.array(row)).collect())
                )

                comb_sigma1 = comb_sigmas[0]
                comb_sigma2 = comb_sigmas[1]

                for feat in split_clus:
                    comb_cov1 = np.dot(r_eigvecs[:, 0], split_corrs[feat].values.T)
                    comb_cov2 = np.dot(r_eigvecs[:, 1], split_corrs[feat].values.T)

                    corr_pc1 = comb_cov1 / comb_sigma1
                    corr_pc2 = comb_cov2 / comb_sigma2

                    if abs(corr_pc1) > abs(corr_pc2):
                        clus1.append(feat)
                    else:
                        clus2.append(feat)

                fin_clus1, fin_clus2, _ = self._reassign_rs(
                    self.df, clus1, clus2, self.n_rs
                )

                c1_eigvals, c1_eigvecs, _, c1_varprops = self.correig(
                    self.df.select(fin_clus1)
                )
                c2_eigvals, c2_eigvecs, _, c2_varprops = self.correig(
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

    def _rsquarespark(self):
        """
        After variable clustering is done, this function calculates the square correlation of each feature with
        (1) its own cluster (2) the "nearest cluster" (3) RS-ratio using own cluster and "nearest clutser"
        RS_Own: Squared correlation between variable and its own cluster
        RS_NC: The largest squared correlation among correlations between variable and all other clusters
        RS_Ratio: (1-RS_Own)/(1-RS_NC)
        Returns
        -------
        rs_table
            Pandas dataframe [Cluster, Variable, RS_Own, RS_NC, RS_Ratio]
            Cluster: integer-type column starting from 0 to maximum number of clusters
            Variable: string-type column. Each row represents a feature name
            RS_Own: float-type column indicating the squared correlation between Variable and its Cluster
            RS_NC: float-type column indicating the squared correlation between Variable and its "nearest cluster"
            RS_Ratio: float-type column (1-RS_Own)/(1-RS_NC)
        """

        cols = ["Cluster", "Variable", "RS_Own", "RS_NC", "RS_Ratio"]
        rs_table = pd.DataFrame(columns=cols)

        sigmas = []
        for _, clusinfo in self.clusters.items():
            c_eigvec = clusinfo.eigvecs[:, 0]
            c_sigma = math.sqrt(
                np.dot(
                    np.dot(
                        c_eigvec, self.corr_df.loc[clusinfo.clus, clusinfo.clus].values
                    ),
                    c_eigvec.T,
                )
            )
            sigmas.append(c_sigma)

        n_row = 0
        for i, clus_own in self.clusters.items():
            for feat in clus_own.clus:
                row = [i, feat]
                cov_own = np.dot(
                    clus_own.eigvecs[:, 0],
                    self.corr_df.loc[feat, clus_own.clus].values.T,
                )

                if len(clus_own.clus) == 1 and feat == clus_own.clus[0]:
                    rs_own = 1
                else:
                    rs_own = (cov_own / sigmas[i]) ** 2

                rs_others = []
                for j, clus_other in self.clusters.items():
                    if j == i:
                        continue

                    cov_other = np.dot(
                        clus_other.eigvecs[:, 0],
                        self.corr_df.loc[feat, clus_other.clus].values.T,
                    )
                    rs = (cov_other / sigmas[j]) ** 2

                    rs_others.append(rs)

                rs_nc = max(rs_others) if len(rs_others) > 0 else 0
                row += [rs_own, rs_nc, (1 - rs_own) / (1 - rs_nc)]
                rs_table.loc[n_row] = row
                n_row += 1

        return rs_table
