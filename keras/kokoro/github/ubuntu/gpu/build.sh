# set -e
set -x

cd "${KOKORO_ROOT}/"

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

PYTHON_BINARY="/usr/bin/python3.9"

"${PYTHON_BINARY}" -m venv venv
source venv/bin/activate
# Check the python version
python --version
python3 --version

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:"
# Check cuda
nvidia-smi
nvcc --version

cd "src/github/keras"
pip install -U pip setuptools

export KERAS_BACKEND="torch"
if [ "$KERAS_BACKEND" == "tensorflow" ]
then
   echo "TensorFlow backend detected."
   pip install -r requirements-tensorflow-cuda.txt --progress-bar off
   pip uninstall -y keras keras-nightly
   echo "Check that TensorFlow uses GPU"
   python3 -c 'import tensorflow as tf;print(tf.__version__);print(tf.config.list_physical_devices("GPU"))'
   # Raise error if GPU is not detected.
   python3 -c 'import tensorflow as tf;assert len(tf.config.list_physical_devices("GPU")) > 0'

   # TODO: keras/layers/merging/merging_test.py::MergingLayersTest::test_sparse_dot_2d Fatal Python error: Aborted
   pytest keras --ignore keras/applications \
               --ignore keras/layers/merging/merging_test.py \
               --cov=keras
fi
deactivate

# TODO: Add test for JAX
if [ "$KERAS_BACKEND" == "jax" ]
then
   echo "JAX backend detected."
   "${PYTHON_BINARY}" -m venv venv1
   source venv1/bin/activate
   pip install -U pip setuptools
   pip install -r requirements-jax-cuda.txt --progress-bar off
   pip uninstall -y keras keras-nightly
   python3 -c 'import jax;print(jax.__version__);print(jax.default_backend())'
   # Raise error if GPU is not detected.
   python3 -c 'import jax;assert jax.default_backend().lower() == "gpu"'

   pytest keras --ignore keras/applications
   deactivate
fi

# TODO: Add test for PyTorch
if [ "$KERAS_BACKEND" == "torch" ]
then
   echo "PyTorch backend detected."
   "${PYTHON_BINARY}" -m venv venv2
   source venv2/bin/activate
   pip install -U pip setuptools
   pip install -r requirements-torch-cuda.txt --progress-bar off
   pip uninstall -y keras keras-nightly
   python3 -c 'import torch;print(torch.__version__);print(torch.cuda.is_available())'
   # Raise error if GPU is not detected.
   python3 -c 'import jax;assert torch.cuda.is_available()'

   pytest keras --ignore keras/applications
   deactivate
fi
