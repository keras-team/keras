FROM nvidia/cuda:7.5-cudnn4-devel

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN mkdir -p $CONDA_DIR && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh && \
    apt-get update && \
    apt-get install -y wget && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-3.9.1-Linux-x86_64.sh && \
    echo "6c6b44acdd0bc4229377ee10d52c8ac6160c336d9cdd669db7371aa9344e1ac3 *Miniconda3-3.9.1-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-3.9.1-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-3.9.1-Linux-x86_64.sh

ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown keras $CONDA_DIR -R && \
    mkdir -p /src && \
    chown keras /src

RUN apt-get install -y g++  # Required for theano to execute optimized C-implementations (for both CPU and GPU)

USER keras

# Python 3.5
#TODO: Add tensorflow
RUN conda install -y python=3.5 numpy scikit-learn notebook pandas matplotlib nose pyyaml six h5py && \
    pip install theano ipdb pytest pytest-cov python-coveralls pytest-xdist pep8 pytest-pep8 && \
    conda clean -yt

ENV THEANO_FLAGS='mode=FAST_RUN,device=gpu,nvcc.fastmath=True,floatX=float32'
ENV PYTHONPATH='/src/keras:$PYTHONPATH'

WORKDIR /src

EXPOSE 8888

CMD jupyter notebook --port=8888 --ip=0.0.0.0

