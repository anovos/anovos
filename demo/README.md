# Anovos Spark Demo: Running workloads with Spark Submit

After checking out via Git clone, please follow the below instructions to run the E2E Anovos Package on the sample
income dataset:

1. First execute the following command to clean the base folder, run unit tests and prepare the latest modules package:

```
make clean build test
```

2. If there is already a working environment with spark, the Demo can be run via the User's local environment directly (
   Note: version dependencies need to be ensured by user) <br>
   For other environments, the Demo can be run using dockers

### via User's local environment

1. Check the pre-requisites - ANOVOS requires Spark (2.4.x), Python (3.7.*), Java(8). Check version using the following
   commands:

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

Once the run has completed, the script will automatically open the final generated
report `report_stats/ml_anovos_report.html` on the browser.

### via Docker

### Build and run Docker image

Note: Kindly ensure the machine has ~15 GB free space atleast when running using Dockers

1. Install docker on your machine (https://docs.docker.com/get-docker/)
2. Set docker settings to use atleast 8GB memory and 4+ cores. Below image shows setting docker settings on Docker
   Desktop:

<p align="center">
  <img src="https://mobilewalla-anovos.s3.amazonaws.com/images/docker_desktop_settings.png" width="800px" title="Docker Desktop Settings">
</p>

3. Ensure dockers is successfully installed by executing the following commands to check docker image and docker
   container respectively:

```
docker image ls
docker ps
```

4. Create docker image and run E2E via Spark using the following command: (Note: Step #1 should have copied a "
   Dockerfile" and "create_anovos_spark_demo_image.sh" to the base directory)

```
./run_anovos_demo.sh
```

5. Once the run has completed, the script will automatically open the final generated report `ml_anovos_report.html` on
   the browser.

### Pull from Docker hub

1. Alternatively, you can choose to pull the demo image from our docker hub public repo and run the image:

```
docker pull anovos/anovos-spark-demo:0.1
docker run --name anovos_demo -t -i -v $(PWD):/temp anovos/anovos-spark-demo:0.1
```

2. Once run has completed, copy the final report (ml_anovos_report.html) from docker container to local and open in
   browser:

```
docker cp anovos_demo:/report_stats/ml_anovos_report.html .
```
