<p align="center">
  <img src="https://mobilewalla-anovos.s3.amazonaws.com/images/anovos.png" width="300px" alt="Anovos">
</p>

<!--
[![Build Status](https://travis-ci.org/mara/mara-pipelines.svg?branch=master)](https://travis-ci.org/mara/mara-pipelines)
[![PyPI - License](https://img.shields.io/pypi/l/mara-pipelines.svg)](https://github.com/mara/mara-pipelines/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/mara-pipelines.svg)](https://badge.fury.io/py/mara-pipelines)
[![Slack Status](https://img.shields.io/badge/slack-join_chat-white.svg?logo=slack&style=social)](https://communityinviter.com/apps/mara-users/public-invite)
-->

Anovos is a an open source project, built by data scientists for the data science community, that brings automation to the feature engineering process. 

By rethinking ingestion and transformation, and including deeper analytics, drift identification, and stability analysis, Anovos improves productivity and helps data scientists build more resilient, higher performing models.

# Installation

### via public registry

To install ANOVOS package via pypi, please execute the following command:  

```
pip3 install anovos
```

### via Git

To install ANOVOS package via git, please execute the following command:  

```
pip3 install "git+https://github.com/anovos/anovos.git"
```

# Architecture

Following diagram shows End to end workflow of ANOVOS. 

<p align="center">
  <img src="https://mobilewalla-anovos.s3.amazonaws.com/images/anovos_architecture.png" width="800px" alt="Anovos Architecture Diagram">
</p>

# Getting Started

Once installed, packages can be imported and the required functionality can be called in the user flow in any application or notebook. <br/>
Refer to the following links to get started:

- [quick-start guide](https://github.com/anovos/anovos/blob/main/notebooks/anovos_quickstart_guide.ipynb)
- [example notebooks](https://github.com/anovos/anovos/tree/main/notebooks)
- [documentation](https://docs.anovos.ai)


# ANOVOS Demo

### Running ANOVOS with Spark Submit

After checking out via Git clone, please follow the below instructions to run the E2E Anovos Package on the sample income dataset: 

1. First execute the following command to clean the base folder, run unit tests and prepare the latest modules package: 
	
```
make clean build test
```

2. If there is already a working environment with spark, the Demo can be run via the User's local environment directly (Note: version dependencies need to be ensured by user) <br>
For other environments, the Demo can be run using dockers

#### via User's local environment

1. Check the pre-requisites - ANOVOS requires Spark (2.4.x), Python (3.7.*), Java(8). Check version using the following commands: 
```
spark-submit --version
python --version
java -version
```
2. Set environment variables - `$JAVA_HOME`, `$SPARK_HOME`, `$PYSPARK_PYTHON`, and `$PATH`
3. Ensure spark-submit and pyspark is working without any issues.
4. Execute the following commands to run the end to end pipeline: 

```
cd dist/
nohup ./spark-submit.sh > run.txt &
```

5. Check result of end to end run

```
tail -f run.txt
```

Once the run has completed, the script will automatically open the final generated report `report_stats/ml_anovos_report.html` on the browser.

#### via Docker

Note: Kindly ensure the machine has ~15 GB free space atleast when running using Dockers
1. Install docker on your machine (https://docs.docker.com/get-docker/)
2. Set docker settings to use atleast 8GB memory and 4+ cores. Below image shows setting docker settings on Docker Desktop:

<p align="center">
  <img src="https://mobilewalla-anovos.s3.amazonaws.com/images/docker_desktop_settings.png" width="800px" title="Docker Desktop Settings">
</p>

3. Ensure dockers is successfully installed by executing the following commands to check docker image and docker container respectively:
```
docker image ls
docker ps
```

4. Create docker image and run E2E via Spark using the following command: (Note: Step #1 should have copied a "Dockerfile_spark_demo" and "create_anovos_spark_demo_image.sh" to the base directory)
	
```
./create_anovos_spark_demo_image.sh
```

5. Once the run has completed, the script will automatically open the final generated report `ml_anovos_report.html` on the browser.

### Running ANOVOS in Notebook

To run and explore ANOVOS in notebooks, if you already have jupyter environment setup, then refer to the [example-notebooks](https://github.com/anovos/anovos/tree/main/notebooks) directly to run.

To run ANOVOS in notebooks via docker, 

1. Run the following command to create the required docker images to run notebook in a container:
```
./create_anovos_notebook_demo_image.sh
```

2. When prompted, open notebook in browser with the prompted URL. For example: http://127.0.0.1:8888/?token=a20b53a2c4e0814a7a7def9e462039beec5b1954e930f7c0 
(Note: As it is running inside container, URL has to be copied and opened in browser manually)

<p align="center">
  <img src="https://mobilewalla-anovos.s3.amazonaws.com/images/docker_notebook_URL.png" width="800px" title="Docker Notebook URL">
</p>

# Running on Custom Dataset

The above E2E ANOVOS package run executed for the sample income dataset can be customized and run for any user dataset. <br>

This can be done configuring the [config/configs.yaml](https://github.com/anovos/anovos/blob/main/config/configs.yaml) file and by defining the flow of run in [src/main/main.py](https://github.com/anovos/anovos/blob/main/src/main/main.py) file.

Kindly refer config files created for the [sales dataset](https://github.com/anovos/anovos/blob/main/config/configs_sales_supervised.yaml) and [segmentation dataset](https://github.com/anovos/anovos/blob/main/config/configs_segmentation_unsupervised.yaml) for reference.

# Documentation

Please find the detailed documentation of ANOVOS [here](https://docs.anovos.ai).

# Reference Links

1. [Setting up pyspark and installing on Windows](https://towardsdatascience.com/installing-apache-pyspark-on-windows-10-f5f0c506bea1)


