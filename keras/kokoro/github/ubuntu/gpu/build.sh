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
   pip install -U tf-nightly
   echo "Check that TensorFlow uses GPU"
   python3 -c 'import tensorflow as tf;print(tf.config.list_physical_devices("GPU"))'
fi
pip uninstall -y keras-nightly

pytest keras --ignore keras/applications --cov=keras
