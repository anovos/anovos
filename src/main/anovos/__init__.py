"""Anovos modules reflect the key components of the Machine Learning (ML) pipeline and are scalable using python API
of Spark (PySpark) - the distributed computing framework.

The key modules included in the alpha release are:

1. **Data Ingest**: This module is an ETL (Extract, transform, load) component of Anovos and helps load dataset(s) as
Spark Dataframe. It also allows performing some basic pre-processing, like selecting, deleting, renaming,
and recasting columns to ensure cleaner data is used in downstream data analysis.

2. **Data Analyzer**: This data analysis module gives a 360º view of the ingested data. It helps provide a better
understanding of the data quality and the transformations required for the modeling purpose. There are three
submodules of this module targeting specific needs of the data analysis process.

    a. *Statistics Generator*: This submodule generates all descriptive statistics related to the ingested data. The
    descriptive statistics are further broken down into different metric types such as Measures of Counts,
    Measures of Central Tendency, Measures of Cardinality, Measures of Dispersion (aka Measures of Spread in
    Statistics), Measures of Percentiles (aka Measures of Position), and Measures of Shape (aka Measures of Moments).

    b. *Quality Checker*: This submodule focuses on assessing the data quality at both row and column levels. It
    includes an option to fix identified issues with the correct treatment method. The row-level quality checks
    include duplicate detection and null detection (% columns that are missing for a row). The column level quality
    checks include outlier detection, null detection (% rows which are missing for a column), biasedness detection (
    checking if a column is biased towards one specific value), cardinality detection (checking if a
    categorical/discrete column have very high no. of unique values) and invalid entries detection which checks for
    suspicious patterns in the column values.

    c. *Association Evaluator*: This submodule focuses on understanding the interaction between different attributes
    (correlation, variable clustering) and/or the relationship between an attribute & the binary target variable (
    Information Gain, Information Value).

3. **Data Drift & Data Stability Computation**: In an ML context, data drift is the change in the distribution of the
baseline dataset that trained the model (source distribution) and the ingested data (target distribution) that makes
the prediction. Data drift is one of the primary causes of poor performance of ML models over time. This module
ensures the stability of the ingested dataset over time by analyzing it with the baseline dataset (via computing
drift statistics) and/or with historically ingested datasets (via computing stability index for existing attributes or
estimating for newly composed features – currently supports only numerical features), if available. Identifying the
data drift at an early stage enables data scientists to be proactive and fix the root cause.

4. **Data Transformer**: In the alpha release, the data transformer module only includes some basic pre-processing
functions like binning, encoding, to name a few. These functions were required to support computations of the above
key modules.  A more exhaustive set of transformations can be expected in future releases.

5. **Data Report**: This module is a visualization component of Anovos. All the analysis on the key modules is
visualized via an HTML report to get a well-rounded understanding of the ingested dataset. The report contains an
executive summary, wiki for data dictionary & metric dictionary, a tab corresponding to key modules demonstrating the
output.

Note: Upcoming Modules - Feature Wiki, Feature store, Auto ML, ML Flow Integration
"""
from .version import __version__
