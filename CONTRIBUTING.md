## How to contribute code

Follow these steps to submit your code contribution.

### Step 1. Open an issue

Before making any changes, we recommend opening an issue (if one doesn't already
exist) and discussing your proposed changes. This way, we can give you feedback
and validate the proposed changes.

If the changes are minor (simple bug fix or documentation fix), then feel free
to open a PR without discussion.

### Step 2. Make code changes

To make code changes, you need to fork the repository. You will need to setup a
development environment and run the unit tests. This is covered in the section
"Setup environment".

### Step 3. Create a pull request

Once the change is ready, open a pull request from your branch in your fork to
the master branch in [keras-team/keras](https://github.com/keras-team/keras).

### Step 4. Sign the Contributor License Agreement

After creating the pull request, the `google-cla` bot will comment on your pull
request with instructions on signing the Contributor License Agreement (CLA) if
you haven't done so. Please follow the instructions to sign the CLA. A `cla:yes`
tag is then added to the pull request.

![Tag added](https://i.imgur.com/LHEdIfL.png)


### Step 5. Code review

A reviewer will review the pull request and provide comments. The reviewer may
add a `kokoro:force-run` label to trigger the continuous integration tests.

![CI tests tag](https://i.imgur.com/58NOCB0.png)

If the tests fail, look into the error messages and try to fix them.

![CI tests](https://i.imgur.com/vVY0dZD.png)

There may be
several rounds of comments and code changes before the pull request gets
approved by the reviewer.

![Approval from reviewer](https://i.imgur.com/Ywl4ets.png)

### Step 6. Merging

Once the pull request is approved, a `ready to pull` tag will be added to the
pull request. A team member will take care of the merging.

![Ready to pull](https://i.imgur.com/yCEqJsA.png)

Here is an [example pull request](https://github.com/keras-team/keras/pull/15015)
for your reference.

## Setup environment

To setup the development environment, We provide two options. One is to use our
Dockerfile, which builds into a container the required dev tools. Another one is
to setup a local environment by installing the dev tools needed.

### Option 1: Use a Docker container

We provide a
[Dockerfile](https://github.com/keras-team/keras/blob/master/.devcontainer/Dockerfile)
to build the dev environment. You can build the Dockerfile into a Docker image
named `keras-dev` with the following command at the root directory of your
cloned repo.

```shell
docker build -t keras-dev .devcontainer
```

You can launch a Docker container from the image with the following command. The
`-it` option gives you an interactive shell of the container. The `-v
path/to/repo/:/home/keras/` mounts your cloned repo to the container. Replace
`path/to/repo` with the path to your cloned repo directory.

```shell
docker run -it -v path/to/repo/:/home/keras/ keras-dev
```

In the container shell, you need to install the latest dependencies with the
following command.

```shell
pip install -r /home/keras/requirements.txt && pip uninstall keras-nightly -y
```

Now, the environment setup is complete. You are ready to run the tests.

You may modify the Dockerfile to your specific needs, like installing your own
dev tools. You may also mount more volumes with the `-v` option, like your SSH
credentials.

Many popular editors today support developing in a container. Here is the list of
[supported editors](https://discuss.tensorflow.org/t/setup-your-favorite-editor-to-develop-keras)
with setup instructions.

### Option 2: Setup a local environment

To setup your local dev environment, you will need the following tools.

1.  [Bazel](https://bazel.build/) is the tool to build and test Keras. See the
    [installation guide](https://docs.bazel.build/versions/4.0.0/install.html)
    for how to install and config bazel for your local environment.
2.  [git](https://github.com/) for code repository management.
3.  [python](https://www.python.org/) to build and code in Keras.

The following commands check the tools above are successfully installed. Note
that Keras requires at least Python 3.7 to run.

```shell
bazel --version
git --version
python --version
```

A [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
(venv) is a powerful tool to create a self-contained environment that isolates
any change from the system level config. It is highly recommended to avoid any
unexpected dependency or version issues.

With the following commands, you create a new venv, named `venv_dir`.

```shell
mkdir venv_dir
python3 -m venv venv_dir
```

You can activate the venv with the following command. You should always run the
tests with the venv activated. You need to activate the venv every time you open
a new shell.

```shell
source venv_dir/bin/activate  # for Linux or MacOS
venv_dir\Scripts\activate.bat  # for Windows
```

Clone your forked repo to your local machine. Go to the cloned directory to
install the dependencies into the venv. Since `tf-nightly` uses `keras-nightly`
as a dependency, we need to uninstall `keras-nightly` so that tests will run
against Keras code in the local workspace.

```shell
git clone https://github.com/YOUR_GITHUB_USERNAME/keras.git
cd keras
pip install -r requirements.txt
pip uninstall keras-nightly
```

The environment setup is completed. You may need to update the `tf-nightly`
version regularly to keep your environment up-to-date with the following
command.

```shell
pip install --upgrade tf-nightly
```

## Code style

The Keras uses [Black](https://black.readthedocs.io/en/stable/) and
[isort](https://pycqa.github.io/isort/) to format the code. Please refer to
[requirements.txt](https://github.com/keras-team/keras/blob/master/requirements.txt)
for the required versions. Run the following command **at the root directory of
the repo** to format your code.

```
sh shell/format.sh
```

It will also display the errors that cannot be resolved by autoformatting. You
need to follow the output of the command to resolve them manually.

If you do not want to auto format the code but only show the lint errors, you
can run `sh shell/lint.sh` **at the root directory of the repo**.

### Docstrings

We do not have an automated way to check docstring style, so if you write
or edit any docstring, please make sure to check them manually.
Keras docstrings follow the conventions below:

A **class docstring** may contain the following items:

* A one-line description of the class.
* Paragraph(s) of more detailed information.
* Optional `Examples` section.
* `Args` section for arguments in `__init__()`.
* If it's a layer:
    * `Call arguments` section for arguments in `Layer.call()`.
    * `Returns` section for the return values of `Layer.call()`.
    * Optional `Raises` section for possible errors.

You can check out `MultiHeadAttention` as an example
[(link)](https://github.com/keras-team/keras/blob/v2.12.0-rc1/keras/layers/attention/multi_head_attention.py#L131).

A **function docstring** may contain the following items:

* One-line description of the function.
* Paragraph(s) of more detailed information.
* Optional `Examples` section.
* `Args` section for the function arguments.
* `Returns` section for the return values.
* Optional `Raises` section for possible errors.

You can check out `text_dataset_from_directory` as an example
[(link)](https://github.com/keras-team/keras/blob/v2.12.0-rc1/keras/utils/text_dataset.py#L31).


## Run tests

We use [Bazel](https://bazel.build/) to build and run the tests.

### Run a test file

For example, to run the tests in `keras/engine/base_layer_test.py`,
we can run the following command at the root directory of the repo.

```shell
bazel test keras/engine:base_layer_test
```

`keras/engine` is the relative path to the directory containing the `BUILD` file
defining the test. `base_layer_test` is the test target name defined with
`tf_py_test` in the `BUILD` file.

### Run a single test case

To run a single test, you can use `--test_filter=<your_regex>`
to use the regular expression to match the test you want to run. For example, you
can use the following command to run all the tests in `activations_test.py`,
whose names contain `test_serialization`.

```
bazel test keras:activations_test --test_filter=*test_serialization*
```

### Run all tests

You can run all the tests locally by running the following command in the repo
root directory.

```
bazel test --test_timeout 300,450,1200,3600 --test_output=errors --keep_going --define=use_fast_cpp_protos=false --build_tests_only --build_tag_filters=-no_oss,-oss_excluded --test_tag_filters=-no_oss,-oss_excluded keras/...
```

### Useful configs

Here we provide a list of useful configs you can use with Bazel.

```shell
bazel test [CONFIGS] [YOUR_TEST]
```

To use these configs, just replace `[CONFIGS]` with the actual config in the
command above.
* `-c opt` enables the optimizations during the build.
* `--test_sharding_strategy=disabled` disables the sharding so that all the
  test outputs are in one file.
  However, it may slow down the tests for not running in parallel
  and may cause the test to timeout.

## Contributing to Keras applications

Contributions to the
[pre-trained application library](https://keras.io/api/applications/) are
welcome. Code for Keras applications is located in Keras repository in
[keras/applications](https://github.com/keras-team/keras/blob/master/keras/applications).
When contributing to Keras applications, please keep following checklist in
mind.

-   Keras applications must implement an established and widely used model.
    Applications should include a link to a paper describing the architecture of
    the model with at least 20 citations.
-   Applications should be provided with pre-trained weights.
    -   When submitting a pull request for a Keras application, these weights
        can be provided at any publically available URL (e.g. a personal Cloud
        Storage bucket). The weights will be uploaded to a Keras storage bucket
        while merging the pull request.
    -   Weights should be downloaded with the
        [get_file()](https://keras.io/api/utils/python_utils/#getfile-function)
        utility function. Be sure to include the `file_hash` argument, which
        allows cache invalidation on the downloaded weights. The command line
        programs `shasum` and `sha256sum` can compute a file hash.
-   You should help us verify that the accuracy of the model with pre-trained
    weighted matches the reported results of the cited paper.
-   You should add any new applications to the unit tests defined in
    `applications_test.py` and `applications_load_weight_test.py`.
-   For backwards compatibility, all applications should provide a
    `preprocess_input()` function. For new applications, you should leave the
    function empty (pass through inputs unaltered), and write the model so it
    can handle raw inputs directly. Adding
    [preprocessing layers](https://keras.io/guides/preprocessing_layers/) to the
    application model may help with this. For image applications, a
    [Rescaling](https://keras.io/api/layers/preprocessing_layers/image_preprocessing/rescaling/)
    layer at the beginning of the model is often all that is needed.
-   Once the PR is approved, you should create a companion PR to the keras.io
    [application page](https://keras.io/api/applications/) updating the
    "Available Models" section. The contribution guide for keras.io can be found
    [here](https://github.com/keras-team/keras-io/blob/master/contributor_guide.md).
-   As every PR requires several CPU/GPU hours of CI testing, we discourage
    submitting PRs to fix one typo, one warning,etc. We recommend fixing the
    same issue at the file level at least (e.g.: fix all typos in a file, fix
    all compiler warnings in a file, etc.)

## Security vulnerability reports

Since Keras is the high-level API of TensorFlow 2, Keras follows same security practices as TensorFlow.
For details on guidelines on vulnerabilities and reporting them, you can refer [Using TensorFlow Securely](https://github.com/tensorflow/tensorflow/blob/master/SECURITY.md). 
