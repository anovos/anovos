ARG spark_version=2.4.8

FROM anovos/anovos-notebook-${spark_version}

WORKDIR /
USER root

# As soon as anovos is available through PyPI, the following two lines can be replaced by "RUN pip install anovos"
# or anovos can be added to the requirements.txt file
RUN pip install git+https://github.com/anovos/anovos.git

# Install additional requirements for provided Jupyter notebooks
COPY requirements.txt /
RUN pip install -r requirements.txt

RUN mkdir anovos && mkdir anovos/data && mkdir anovos/notebooks && mkdir anovos/guides
COPY data/ /anovos/data
COPY notebooks/ /anovos/notebooks
COPY guides/ /anovos/guides

# Ensure that the anovos folder is writeable
RUN fix-permissions /anovos

CMD start-notebook.sh --NotebookApp.notebook_dir=/anovos
