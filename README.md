<p align="center">
  <img src="https://mobilewalla-anovos.s3.amazonaws.com/anovos.png" width="800" title="Anovos Logo">
</p>

# Steps to Install ANOVOS

## Installation via public registry

To install ANOVOS package via pypi, please execute the following commands:  

`pip3 install anovos`


## Installation via Git

To install ANOVOS package via git, please execute the following commands:  

`pip3 install "git+https://gitlab.com/mwengr/mw_ds_feature_machine.git"`


# Get Started

Once installed, packages can be imported and the required functionality can be called in the user flow in any application or notebook. <br/>
Refer to the following links to get started :

- quick-start guide
- example notebooks
- documentation

# Steps to Run ANOVOS using spark-submit

After checking out via Git clone, please follow the below instructions to run the E2E ML Anovos Package on the sample income dataset: 

1. First execute the following command to clean folder, build the latest modules: 
	
	`make clean build`

2. There are 2 ways to run using spark-submit after this:

- Follow A, if you have a working environment already and would just like to use the same configs. (Note: version dependencies are to be ensured by user)
- Follow B, if you want to run via dockers

## A. Running via User's local environment

1. Check the pre-requisites - ANOVOS requires Spark (2.4.x), Python (3.7.*), Java(8). Check version using the following commands: 

	`spark-submit --version`<br/>
	`python --version`<br/>
	`java -version`

2. Set environment variables - `$JAVA_HOME`, `$SPARK_HOME`, `$PYSPARK_PYTHON`, and `$PATH`
3. Ensure spark-submit and pyspark is working without any issues.
4. Execute the following commands to run the end to end pipeline: 

	`cd dist/`<br/>
	`nohup ./spark-submit.sh > run.txt &`

5. Check result of end to end run

	`tail -f run.txt`

## B. Running via Dockers

Note: Kindly ensure the machine has ~15 GB free space atleast when running using Dockers

1. Install docker on your machine (https://docs.docker.com/get-docker/)
2. Set docker settings to use atleast 8GB memory and 4+ cores. Below image shows setting docker settings on Docker Desktop:

<p align="center">
  <img src="figures/docker_desktop_settings.png" width="800" title="Docker Desktop Settings">
</p>

3. Ensure dockers is successfully installed by executing the following commands to check docker image and docker container respectively:
	`docker image ls`<br>
	`docker ps`

4. Create docker image with the following command: (Note: Step #1 should have copied a "Dockerfile" to the directory the following command is being executed in)
	
	`docker build -t mw_ds_feature_machine:0.1 .`

5. Check if docker image was successfully created: 

	`docker image ls`

6. Run the docker container using the following command:

	`docker run -t -i -v $(PWD):/temp mw_ds_feature_machine:0.1`

7. Check if docker container is successfully running: 

	`docker ps -a`

8. To explore the generated output folder, execute the following commands:

	`docker exec -it <container_id> bash`

9. Once run has completed please exit the docker run by sending SIGINT - `^C` (CTRL + C)


# Running on custom dataset

The above E2E ANOVOS package run executed for the sample income dataset can be customized and run for any user dataset. <br>
This can be done configuring the `config/config.yaml` file and by defining the flow of run in `src/main.py` file.

# ANOVOS Package Documentation

Please find the detailed documentation of ANOVOS package here.

# Reference Links

1. Setting up pyspark and installing on Windows <br>
https://towardsdatascience.com/installing-apache-pyspark-on-windows-10-f5f0c506bea1


