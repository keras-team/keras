---
name: Bug Issue
about: Use this template for reporting a bug
labels: 'type:bug'

---

Please go to TF Forum for help and support:

https://discuss.tensorflow.org/tag/keras

If you open a GitHub issue, here is our policy:

It must be a bug, a feature request, or a significant problem with the documentation (for small docs fixes please send a PR instead).
The form below must be filled out.

**Here's why we have that policy:**.

Keras developers respond to issues. We want to focus on work that benefits the whole community, e.g., fixing bugs and adding features. Support only helps individuals. GitHub also notifies thousands of people when issues are filed. We want them to see you communicating an interesting problem, rather than being redirected to Stack Overflow.

**System information**.
- Have I written custom code (as opposed to using a stock example script provided in Keras):
- OS Platform and Distribution (e.g., Linux Ubuntu 16.04):
- TensorFlow installed from (source or binary):
- TensorFlow version (use command below):
- Python version:
- Bazel version (if compiling from source):
- GPU model and memory:
- Exact command to reproduce:

You can collect some of this information using our environment capture script:

https://github.com/tensorflow/tensorflow/tree/master/tools/tf_env_collect.sh

You can obtain the TensorFlow version with:
python -c "import tensorflow as tf; print(tf.version.GIT_VERSION, tf.version.VERSION)"

**Describe the problem**.

Describe the problem clearly here. Be sure to convey here why it's a bug in Keras or why the requested feature is needed.

**Describe the current behavior**.


**Describe the expected behavior**.

**[Contributing](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md)**.

- Do you want to contribute a PR? (yes/no):
- If yes, please read [this page](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md) for instructions
- Briefly describe your candidate solution(if contributing):

**Standalone code to reproduce the issue**.

Provide a reproducible test case that is the bare minimum necessary to generate
the problem. If possible, please share a link to Colab/Jupyter/any notebook.


**Source code / logs**.

Include any logs or source code that would be helpful to diagnose the problem. If including tracebacks, please include the full traceback. Large logs and files should be attached. Try to provide a reproducible test case that is the bare minimum necessary to generate the problem.
