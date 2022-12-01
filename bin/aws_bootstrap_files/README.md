# AWS Bootstrap Files

## To use ANOVOS on AWS EMR:

It is recommended to use EMR version 5.30 or above as it already has Python3 as the default version .There are two ways to install Anovos on AWS environment. See below for further details.

1.Firstly by Installing Anovos dependency packages from Anovos developmental github version

To install Anovos dependency packages seperately, you can copy the [requirements.txt](https://github.com/anovos/anovos/blob/main/requirements.txt) to an accessible s3 bucket, correspondingly edit the first line to use the following file as part of bootstrap actions when starting the cluster - `setup_on_aws_emr_5_30_and_above.sh`

2.Secondly by installing anovos through PyPI (Latest released version)


Kindly use `setup_on_aws_emr_5_30_and_above_latest_release_version.sh` file as part of the bootstrap actions when starting the cluster to directly install anovos (latest release version):

```bash
sudo pip3 install --upgrade cython
sudo pip3 install anovos
```

