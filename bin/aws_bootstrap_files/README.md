# AWS Bootstrap Files

## To use ANOVOS on AWS EMR:

Recommend to use EMR version 5.30 or above as it already has Python3 as the system default.

Kindly include the following line as part of the bootstrap actions sh file when starting the cluster to directly install anovos (latest release version):

```bash
sudo pip3 install anovos
```

To install Anovos dependency packages seperately, kindly copy the [requirements.txt](https://github.com/anovos/anovos/blob/main/requirements.txt) to an accessible s3 bucket, correspondingly edit the first line to use the following file as part of bootstrap actions when starting the cluster - `setup_on_aws_emr_5_30_and_above.sh`
