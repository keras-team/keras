FROM --platform=linux/amd64 python:3.10-slim

ENV KERAS_HOME=/github/workspace/.github/workflows/config/jax \
    KERAS_BACKEND=jax

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire codebase into the container
COPY . /github/workspace
WORKDIR /github/workspace

# Create and activate venv, install pip/setuptools/psutil, then run tests
# RUN cd ./keras/src/github/keras && \
RUN pip install --no-cache-dir -U pip setuptools && \
    pip install --no-cache-dir -U psutil && \
    pip install --no-cache-dir -r requirements-jax-tpu.txt && \
    pip uninstall -y keras keras-nightly && \
    python3 -c 'import jax;print(jax.__version__);print(jax.default_backend())' && \
    python3 -c 'import jax;assert jax.default_backend().lower() == "tpu"' && \
    pytest keras --ignore keras/src/applications \
                 --ignore keras/src/layers/merging/merging_test.py \
                 --cov=keras \
                 --cov-config=pyproject.toml

CMD ["/bin/bash"]
