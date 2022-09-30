ARG spark_version=3.2.1
ARG anovos_version=0.3.0

FROM anovos/anovos-notebook-${spark_version}:${anovos_version}

ARG spark_version
ARG anovos_version

WORKDIR /
USER root

RUN pip install --no-cache-dir "pyspark==${spark_version}" "anovos==${anovos_version}"

ENTRYPOINT ["python", "-m", "anovos", "/config.yaml", "local"]
