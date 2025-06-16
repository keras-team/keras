set -e
set -x

cd "${KOKORO_ROOT}/"

sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

PYTHON_BINARY="/usr/bin/python3.10"

"${PYTHON_BINARY}" -m venv venv
source venv/bin/activate
# Check the python version
python --version
python3 --version

cd "src/github/keras"
pip install -U pip setuptools
# psutil is used by background log reader
pip install -U psutil

if [ "$KERAS_BACKEND" == "tensorflow" ]
then
   echo "TensorFlow backend detected."
   pip install -r requirements-tensorflow-tpu.txt --progress-bar off --timeout 1000
   pip uninstall -y keras keras-nightly
   echo "Check that TensorFlow uses TPU"
   python3 -c 'import tensorflow as tf;print(tf.__version__);print(tf.config.list_physical_devices("TPU"))'
   # Raise error if GPU is not detected.
   python3 -c 'import tensorflow as tf;assert len(tf.config.list_physical_devices("TPU")) > 0'

   # TODO: keras/layers/merging/merging_test.py::MergingLayersTest::test_sparse_dot_2d Fatal Python error: Aborted
   pytest keras --ignore keras/src/applications \
               --ignore keras/src/layers/merging/merging_test.py \
               --cov=keras \
               --cov-config=pyproject.toml
fi