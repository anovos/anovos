# Locally run Anovos workloads without access to a Spark installation/cluster

Using the `run_workload.sh` script, you can launch _Anovos_ workloads defined
in configuration files without having to install Spark or having access to
a Spark cluster.

The following command will run the workflow defined in `config.yaml`
on Spark 3.2.2:

```shell
./run_workload.sh config.yaml 3.2.2
```

Note that all paths to input data in the configuration file have to be given
relative to the directory you are calling `run_workload.sh` from.

The output will be placed in a directory called `output` inside the directory
you are calling `run_workload.sh` from.

For more information, see
[🐋 Running _Anovos_ workloads through Docker](https://docs.anovos.ai/using-anovos/setting-up/locally.html#running-anovos-workloads-through-docker)
in the documentation.

## Build a custom `anovos-worker` image

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

Once you have an `anovos-worker` image, you can launch _Anovos_ workloads
[as described in the documentation](https://docs.anovos.ai/using-anovos/setting-up/locally.html#running-anovos-workloads-through-docker)
but without specifying the Spark version:

```bash
./run_workload.sh config.yaml
```

If your custom image is not called `anovos-worker`, you will need to
adapt `run_workload.sh` accordingly.
