ARG spark_version=3.2.1
ARG image_version=latest

FROM anovos/anovos-notebook-${spark_version}:${image_version}

WORKDIR /
USER root

# As soon as anovos is available through PyPI, the following two lines can be replaced by "RUN pip install anovos"
# or anovos can be added to the requirements.txt file
#RUN pip install git+https://github.com/anovos/anovos.git

# Install additional requirements for provided Jupyter notebooks
COPY requirements.txt /
RUN pip install -r requirements.txt

RUN mkdir anovos && mkdir anovos/use_case_demo
COPY use_case_demo/ /anovos/use_case_demo
RUN wget "https://mobilewalla-anovos.s3.amazonaws.com/workshop/data.tgz" \
         && tar -xzvf data.tgz
RUN mv data /anovos/use_case_demo/data

# Ensure that the anovos folder is writeable
RUN fix-permissions /anovos

CMD start-notebook.sh --NotebookApp.notebook_dir=/anovos --NotebookApp.port=9999
