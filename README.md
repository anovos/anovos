
To run after checking out, please follow the below instructions : 

Run all the below commands from repo home path : mw_ds_feature_machine/. unless states specifically with a changedir (cd) :

1. First execute the following command to clean folder, build the latest modules and run unit tests : 
	make clean build test
This step is to ensure there are no unit test errors and build is successful in the checked out branch.

2. There are 2 ways to run after this :
Follow 2.A if you have a working environment already and would just like to use the same configs. (Note : version dependencies are ensured by the user in this path)
Follow 2.B.if you want to run via dockers

2.A. RUNNING USING USER'S CURRENT ENVIRONMENT :
2.A.1. Set environment variables - $SPARK_HOME, $PYSPARK_PYTHON, and $PATH .
2.A.2. Ensure spark-submit and pyspark is working without any glitches.
2.A.3. Execute the following commands to run the end to end pipeline : 
	cd dist/
	./spark-submit.sh

2.B. RUNNING USING DOCKERS : (Note : Kindly ensure the machine has ~15 GB free space atleast when running using Dockers)
2.B.1. Install dockers on your machine (https://docs.docker.com/get-docker/)
2.B.2. Ensure dockers is successfully installed by executing the following commands to check docker image and docker container respectively :
	docker image ls
	docker ps
2.B.3. Create docker image with the following command : (Note : Step #1 should have copied "Dockerfile" to the directory the following command is executed in)
	docker build -t mw_ds_feature_machine:0.1 .
2.B.4. Check if docker image was successfully created : 
	docker image ls
2.B.5. Run the docker using the following command :
	docker run -t -i -v $(PWD):/temp mw_ds_feature_machine:0.1
2.B.6. Check if docker container is successfully running : 
	docker ps -a
2.B.7. To explore the output folder, execute the following commands :
	docker exec -it <container_id> bash
