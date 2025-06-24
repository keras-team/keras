FROM python:3.10-slim

ENV KERAS_HOME=/github/workspace/.github/workflows/config/tensorflow \
    KERAS_BACKEND=tensorflow

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire codebase into the container
COPY . /github/workspace
WORKDIR /github/workspace

# Create and activate venv, install pip/setuptools/psutil, then run tests
RUN cd src/github/keras && \
    pip install -U pip setuptools && \
    pip install -U psutil && \
    pip install -r requirements-tensorflow-tpu.txt && \
    pip uninstall -y keras keras-nightly && \
    python3 -c 'import tensorflow as tf;print(tf.__version__);print(tf.config.list_physical_devices("TPU"))' && \
    python3 -c 'import tensorflow as tf;assert len(tf.config.list_physical_devices("TPU")) > 0' && \
    pytest keras --ignore keras/src/applications \
                 --ignore keras/src/layers/merging/merging_test.py \
                 --cov=keras \
                 --cov-config=pyproject.toml

CMD ["bash"]