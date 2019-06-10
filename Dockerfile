FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \ 
        build-essential \
        git \
        curl \
        ca-certificates \
        && \
     rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6
ENV NAME=topaz-py$PYTHON_VERSION

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create -y --name $NAME python=$PYTHON_VERSION numpy pandas scikit-learn && \
     /opt/conda/bin/conda clean -ya 

ENV PATH /opt/conda/bin:$PATH
ENV PATH /opt/conda/envs/$NAME/bin:$PATH
RUN conda install --name $NAME -c pytorch pytorch torchvision

# setup topaz install directory
WORKDIR /opt/topaz
COPY . .
# now install
RUN pip install -v .

