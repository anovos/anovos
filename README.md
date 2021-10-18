

# Steps to Run ML ANOVOS Package :

After checking out, please follow the below instructions to run the ML Anovos Package : 

1. First execute the following command to clean folder, build the latest modules : 
	
	`make clean build`

2. There are 2 ways to run after this :

- Follow A, if you have a working environment already and would just like to use the same configs. (Note : version dependencies are to be ensured by user)
- Follow B, if you want to run via dockers

## A. Running via User's local environment :

1. Check the pre-requisites - Anovos requires Spark (2.4.x), Python (3.7+), Java(8)
2. Set environment variables - `$JAVA_HOME`, `$SPARK_HOME`, `$PYSPARK_PYTHON`, and `$PATH`
3. Ensure spark-submit and pyspark is working without any issues.
4. Execute the following commands to run the end to end pipeline : 

	`cd dist/`
	
	`./spark-submit.sh`

## B. Running via Dockers: 

Note : Kindly ensure the machine has ~15 GB free space atleast when running using Dockers

1. Install docker on your machine (https://docs.docker.com/get-docker/)
2. Set docker settings to use atleast 8GB memory and 4+ cores. Below image shows setting docker settings on Docker Desktop :

<p align="center">
  <img src="figures/docker_desktop_settings.png" width="800" title="Docker Desktop Settings">
</p>

3. Ensure dockers is successfully installed by executing the following commands to check docker image and docker container respectively :
	`docker image ls`

	`docker ps`

4. Create docker image with the following command : (Do Note : Step #1 should have copied a "Dockerfile" to the directory the following command is executed in)
	
	`docker build -t mw_ds_feature_machine:0.1 .`

5. Check if docker image was successfully created : 

	`docker image ls`

6. Run the docker container using the following command :

	`docker run -t -i -v $(PWD):/temp mw_ds_feature_machine:0.1`

7. Check if docker container is successfully running : 

	`docker ps -a`

8. To explore the generated output folder, execute the following commands :

	`docker exec -it <container_id> bash`

9. Once run has completed please exit the docker run by sending SIGINT - `^C` (CTRL + C)
