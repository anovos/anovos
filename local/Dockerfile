ARG spark_version=3.2.1
ARG anovos_version=0.3.0

FROM anovos/anovos-notebook-${spark_version}:${anovos_version}

ARG spark_version
ARG anovos_version

WORKDIR /
USER root

# TODO: Why is there no PySpark in the image?
RUN pip install "pyspark==${spark_version}"

RUN pip install "anovos==${anovos_version}"

ENTRYPOINT ["python", "-m", "anovos", "/config.yanl", "local"]
