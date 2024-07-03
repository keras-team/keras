set -e
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
# psutil is used by background log reader
pip install -U psutil

if [ "$KERAS_BACKEND" == "tensorflow" ]
then
   echo "TensorFlow backend detected."
   pip install -r requirements-tensorflow-cuda.txt --progress-bar off --timeout 1000
   pip uninstall -y keras keras-nightly
   echo "Check that TensorFlow uses GPU"
   python3 -c 'import tensorflow as tf;print(tf.__version__);print(tf.config.list_physical_devices("GPU"))'
   # Raise error if GPU is not detected.
   python3 -c 'import tensorflow as tf;assert len(tf.config.list_physical_devices("GPU")) > 0'

   # TODO: keras/layers/merging/merging_test.py::MergingLayersTest::test_sparse_dot_2d Fatal Python error: Aborted
   pytest keras --ignore keras/src/applications \
               --ignore keras/src/layers/merging/merging_test.py \
               --cov=keras
fi

if [ "$KERAS_BACKEND" == "jax" ]
then
   echo "JAX backend detected."
   pip install -r requirements-jax-cuda.txt --progress-bar off --timeout 1000
   pip uninstall -y keras keras-nightly
   python3 -c 'import jax;print(jax.__version__);print(jax.default_backend())'
   # Raise error if GPU is not detected.
   python3 -c 'import jax;assert jax.default_backend().lower() == "gpu"'

   # TODO: keras/layers/merging/merging_test.py::MergingLayersTest::test_sparse_dot_2d Fatal Python error: Aborted
   # TODO: keras/trainers/data_adapters/py_dataset_adapter_test.py::PyDatasetAdapterTest::test_basic_flow0 Fatal Python error: Aborted
   # keras/backend/jax/distribution_lib_test.py is configured for CPU test for now.
   pytest keras --ignore keras/src/applications \
               --ignore keras/src/layers/merging/merging_test.py \
               --ignore keras/src/trainers/data_adapters/py_dataset_adapter_test.py \
               --ignore keras/src/backend/jax/distribution_lib_test.py \
               --ignore keras/src/distribution/distribution_lib_test.py \
               --cov=keras

   pytest keras/src/distribution/distribution_lib_test.py --cov=keras
fi

if [ "$KERAS_BACKEND" == "torch" ]
then
   echo "PyTorch backend detected."
   pip install -r requirements-torch-cuda.txt --progress-bar off --timeout 1000
   pip uninstall -y keras keras-nightly
   python3 -c 'import torch;print(torch.__version__);print(torch.cuda.is_available())'
   # Raise error if GPU is not detected.
   python3 -c 'import torch;assert torch.cuda.is_available()'

   pytest keras --ignore keras/src/applications \
               --cov=keras
fi
