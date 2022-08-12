# Locally run Anovos workloads without access to a Spark installation/cluster

Note that this is currently an experimental feature and details are subject
to change without prior notice.
Once this way of running Anovos workloads is stable, we will publish the
required Docker images on Docker Hub.
For now, you'll need to build the image yourself.

Build the `anovos-worker` image:

```bash
docker build . -t anovos-worker
```

You can specify the version of Apache Spark and Anovos using build args:
```bash
docker build --build-arg spark_version=3.2.1 --build-arg anovos_version=0.3.0 .
```

Note that a corresponding image `anovos/anovos-notebook-${spark_version}`
needs to be available on Docker Hub.
You can check the available images [here](https://hub.docker.com/u/anovos).

If you need a different configuration, you can build your own base image
using the [`build_image.sh`](../examples/anovos_notebook/build_image.sh) script.

Once you have an `anovos-worker` image, you can run Anovos workloads as follows:

```bash
./run_workload.sh config.yaml
```

Note that all paths to input data in the configuration file have to be given
relative to the directory you are calling `run_workload.sh` from.

The output will be placed in a directory called `output` inside the directory
you are calling `run_workload.sh` from.
