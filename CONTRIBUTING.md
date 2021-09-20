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
development environment and run the unit tests. This is covered in section
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

A reviewer will review the pull request and provide comments.
The reviewer may add a `kokoro:force-run` label to trigger the 
continuous integration tests.

![CI tests tag](https://i.imgur.com/58NOCB0.png)

If the tests fail, look into the error messages and try to fix it.

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
to setup a local environment by install the dev tools needed.

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
pip install -r /home/keras/requirements.txt
```

Now, the environment setup is complete. You are ready to run the tests.

You may modify the Dockerfile to your specific needs, like installing your own
dev tools. You may also mount more volumes with the `-v` option, like your SSH
credentials.

Many popular editors today support developing in a container. Here is list of
[supported editors](https://discuss.tensorflow.org/t/setup-your-favorite-editor-to-develop-keras)
with setup instructions.

### Option 2: Setup a local environment

To setup your local dev environment, you will need the following tools.

1.  [Bazel](https://bazel.build/) is the tool to build and test Keras. See the
    [installation guide](https://docs.bazel.build/versions/4.0.0/install.html)
    for how to install and config bazel for your local environment.
2.  [git](https://github.com/) for code repository management.
3.  [python](https://www.python.org/) to build and code in Keras.

The following commands checks the tools above are successfully installed. Note
that Keras requires at least Python 3.7 to run.

```shell
bazel --version
git --version
python --version
```

A [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
(venv) is a powerful tool to create a self-contained environment that isolates
any change from the system level config. It is highly recommended to avoid any
unexpected dependency or version issue.

With the following commands, you create a new venv, named `venv_dir`.

```shell
mkdir venv_dir
python3 -m venv venv_dir
```

You can activate the venv with the following command. You should always run the
tests with the venv activated. You need to activate the venv everytime you open
a new shell.

```shell
source venv_dir/bin/activate  # for linux or MacOS
venv_dir\Scripts\activate.bat  # for Windows
```

Clone your forked repo to your local machine. Go to the cloned directory to
install the dependencies into the venv. Since `tf-nightly` uses `keras-nightly`
as a dependency, we need to uninstall `keras-nightly` so that tests will run
against keras code in local workspace.

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

## Run tests

We use [Bazel](https://bazel.build/) to build and run the tests.

### Run a test file

For example, to run the tests in `keras/engine/base_layer_test.py`,
we can run the following command at the root directory of the repo.

```shell
bazel test keras/engine:base_layer_test
```

`keras/engine` is the relative path to the directory
containing the `BUILD` file defing the test.
`base_layer_test` is the test target name defined  with `tf_py_test`
in the `BUILD` file.

### Run a single test case

The best way to run a single test case is to comment out the rest of the test
cases in a file before runing the test file.

### Run all tests

You can run all the tests locally by running the following commmand
in the repo root directory.

```
bazel test --test_timeout 300,450,1200,3600 --test_output=errors --keep_going --define=use_fast_cpp_protos=false --build_tests_only --build_tag_filters=-no_oss --test_tag_filters=-no_oss keras/...
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
