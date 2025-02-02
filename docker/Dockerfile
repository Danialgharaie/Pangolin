######################################
## CONTAINER FOR GPU based pangolin ##
######################################

# start from the cuda docker base
from nvidia/cuda:12.0.0-runtime-ubuntu22.04

## needed apt packages
ARG BUILD_PACKAGES="wget git bzip2"
# needed conda packages 
ARG CONDA_PACKAGES="python==3.10.8 pip==25.0 pandas==2.2.3 pyfastx==0.8.4 gffutils==0.13 pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda==12.4 pysam==0.20.0"
ARG CONDA_CHANNEL="-c nvidia -c pytorch -c conda-forge -c anaconda -c bioconda"
##  ENV SETTINGS during runtime
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH
ENV DEBIAN_FRONTEND noninteractive

## AUTHOR
ENV AUTHOR="Geert Vandeweyer"
ENV EMAIL="geert.vandeweyer@uza.be"

# For micromamba:
SHELL ["/bin/bash", "-l", "-c"]
ENV MAMBA_ROOT_PREFIX=/opt/conda/
ENV PATH=/opt/micromamba/bin:/opt/conda/bin:$PATH


## INSTALL 
RUN apt-get -y update  && \
    apt-get -y install $BUILD_PACKAGES && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
     
    
# conda packages
RUN mkdir /opt/conda && \
    mkdir /opt/micromamba && \
    wget -qO - https://micromamba.snakepit.net/api/micromamba/linux-64/0.23.0 | tar -xvj -C /opt/micromamba bin/micromamba && \
    # initialize bash
    micromamba shell init --shell=bash --prefix=/opt/conda && \
    # remove a statement from bashrc that prevents initialization
    grep -v '[ -z "\$PS1" ] && return' /root/.bashrc  > /opt/micromamba/bashrc && \
    mv /opt/micromamba/bashrc /root/.bashrc && \
    source ~/.bashrc && \
    # activate & install base conda packag
    micromamba activate && \
    micromamba install -y $CONDA_CHANNEL $CONDA_PACKAGES && \
    micromamba clean --all --yes

# my fork of pangolin : has gpu optimizations
RUN cd /opt/ && \
    git clone https://github.com/geertvandeweyer/pangolin.git && \
    cd pangolin && \
    pip install .  
    
# ADD annotation data
CMD ["pangolin", "--help"]
