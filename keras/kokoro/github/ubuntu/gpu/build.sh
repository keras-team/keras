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
pip install -r requirements.txt --progress-bar off

if [ "$KERAS_BACKEND" == "tensorflow" ]
then
   echo "TensorFlow backend detected."
   pip uninstall -y tensorflow-cpu
   pip uninstall -y keras
   # TF 2.14 is not built with Cuda 12.2 and doesn't detect GPU
   # TODO: Use TF Nightly until TF 2.15 RC is released
   pip install -U tf-nightly
   pip uninstall -y keras-nightly
   echo "Check that TensorFlow uses GPU"
   python3 -c 'import tensorflow as tf;print(tf.__version__);print(tf.config.list_physical_devices("GPU"))'
   # Raise error if GPU is not detected by TensorFlow.
   python3 -c 'import tensorflow as tf;len(tf.config.list_physical_devices("GPU")) > 0'

   # TODO: keras/layers/merging/merging_test.py::MergingLayersTest::test_sparse_dot_2d Fatal Python error: Aborted
   # TODO: Embedding test failure
   # TODO: Backup and Restore fails
   pytest keras --ignore keras/applications \
               --ignore keras/layers/merging/merging_test.py \
               --ignore keras/layers/core/embedding_test.py \
               --ignore keras/callbacks/backup_and_restore_callback_test.py \
               --cov=keras
fi

# TODO: Add test for JAX
if [ "$KERAS_BACKEND" == "jax" ]
then
   echo "JAX backend detected."
fi

# TODO: Add test for PyTorch
if [ "$KERAS_BACKEND" == "torch" ]
then
   echo "PyTorch backend detected."
fi
