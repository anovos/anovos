# AWS Bootstrap Files

## To use ANOVOS on AWS EMR:

### For EMR >= 5.30:
No additional cluster configuration for Python3 is required, as it is already the system default.

Kindly include the following line as part of the bootstrap actions sh file when starting the cluster to directly install anovos (latest release version):

```bash
sudo pip3 install anovos
```

To install Anovos dependency packages seperately, kindly copy the [requirements.txt](https://github.com/anovos/anovos/blob/main/requirements.txt) to an accessible s3 bucket, correspondingly edit the first line to use the following file as part of bootstrap actions when starting the cluster - `setup_on_aws_emr_5_30_and_above.sh`

### For EMR version < 5.30:
Enable Python3 configuration in the EMR cluster configurations as it is not the system default:
```json
[{"configurations":[{"classification":"export","properties":{"PYSPARK_PYTHON":"/usr/bin/python3"}}],"classification":"spark-env","properties":{}}]
```

Kindly include the following line as part of the bootstrap actions sh file when starting the cluster to directly install anovos (latest release version):
```bash
sudo pip3.7 install anovos
```

To install Anovos dependency packages seperately, kindly copy the [requirements.txt](https://github.com/anovos/anovos/blob/main/requirements.txt) to an accessible s3 bucket, correspondingly edit the first line to use the following file as part of bootstrap actions when starting the cluster - `setup_on_aws_emr_below_5_30.sh`
