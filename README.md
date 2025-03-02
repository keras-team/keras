# 🚀 Keras 3: Deep Learning for Humans  

Keras 3 is a **multi-backend deep learning framework**, with support for **JAX, TensorFlow, PyTorch,** and **OpenVINO** (for inference-only).  
Effortlessly build and train models for:  
📸 **Computer Vision** | 📝 **Natural Language Processing** | 🎵 **Audio Processing** | ⏳ **Time Series Forecasting** | ⭐ **Recommender Systems**  

### 🔥 Why Choose Keras 3?  
✅ **Accelerated Model Development** – Ship deep learning solutions **faster** with an intuitive high-level API.  
✅ **State-of-the-Art Performance** – Pick the **fastest backend** for your architecture (often **JAX!** 🚀). [🔗 Benchmark Here](https://keras.io/getting_started/benchmarks/)  
✅ **Datacenter-Scale Training** – Seamlessly scale from **laptop to large GPU/TPU clusters**.  

Join **nearly three million developers** 🌎—from startups to global enterprises—in harnessing the power of **Keras 3**.  

---

## 📥 Installation  

### 🔹 Install with pip  

Keras 3 is available on **PyPI** as `keras`. *(Keras 2 remains available as `tf-keras`.)*  

1️⃣ **Install Keras:**  
```bash
pip install keras --upgrade
```  
2️⃣ **Install Backend Package(s):**  
To use `keras`, install at least one backend: `tensorflow`, `jax`, or `torch`.  
🔹 *TensorFlow is required for some features like preprocessing layers and `tf.data` pipelines.*  

---

## 💻 Local Installation  

### 🏗️ Minimal Installation  

Keras 3 is compatible with **Linux & macOS**.  
🪟 Windows users: Use **WSL2** for best support.  

📌 **Steps to install a local development version:**  

1️⃣ Install dependencies:  
```bash
pip install -r requirements.txt
```  
2️⃣ Run installation command:  
```bash
python pip_build.py --install
```  
3️⃣ Run API generation script when updating public APIs:  
```bash
./shell/api_gen.sh
```  

### ⚡ Adding GPU Support  

The `requirements.txt` installs **CPU-only** versions of TensorFlow, JAX, and PyTorch.  
For **GPU support**, use `requirements-{backend}-cuda.txt` files.  

Example: Setting up a **JAX GPU environment** with `conda`:  
```bash
conda create -y -n keras-jax python=3.10
conda activate keras-jax
pip install -r requirements-jax-cuda.txt
python pip_build.py --install
```  

---

## 🔧 Configuring Your Backend  

🛠️ Set the backend using an **environment variable** or by editing `~/.keras/keras.json`.  
Available backends: `"tensorflow"`, `"jax"`, `"torch"`, `"openvino"`  

📌 Example: Setting JAX as the backend  
```bash
export KERAS_BACKEND="jax"
```  
📌 In **Google Colab**:  
```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
```  

🚨 **Note:** The backend **must be configured before** importing `keras`, and it **cannot** be changed after import.  
🚀 **OpenVINO Backend** is **inference-only** – use it for `model.predict()` tasks.  

---

## 🔄 Backwards Compatibility  

Keras 3 works as a **drop-in replacement** for `tf.keras` *(when using TensorFlow backend)*.  
✔️ **For existing `tf.keras` code** – update your `model.save()` calls to the **`.keras` format**.  
✔️ **Custom components?** Convert them to a **backend-agnostic** version in minutes!  
✔️ **Supports both** `tf.data.Dataset` and PyTorch `DataLoaders`.  

---

## 🌟 Why Use Keras 3?  

💡 **Flexibility:** Run your models on **any framework**—switch backends at will!  
💡 **Future-Proof:** Avoid **framework lock-in** and keep your code **portable**.  
💡 **For PyTorch Users:** Enjoy **the power of Keras** with **PyTorch's flexibility**!  
💡 **For JAX Users:** Get a **fully-featured**, battle-tested, and well-documented **ML library**.  
💡 **For Custom Training Loops:** Use Keras **inside native TF, JAX, or PyTorch workflows**.  

📝 **Read more in the** [Keras 3 release announcement](https://keras.io/keras_3/). 🚀  

