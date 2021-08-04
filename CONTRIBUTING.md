## How to contribute code

You can follow these steps to submit your code contribution.

### Step 1. Open an issue
Before making any changes, we recommend opening an issue (if one doesn't already
exist) and discussing your proposed changes. This way, we can give you feedback
and validated the proposed changes. If the changes are minor (simple bug fix
of documentation fix), then feel free to open a PR without discussion.

### Step 2. Make code changes

To make code changes, you need to fork the repository.
You may need to setup a dev environment,
run the unit tests, which are introduced in the later sections of this guide.

### Step 3. Create a pull request 

Once the code changes are made,
you can open a pull request from your branch in your fork to the master branch in
[keras-team/keras](https://github.com/keras-team/keras).

### Step 4. Sign the agreement

After creating the pull request,
the `google-cla` bot will comment on your pull request
with the instructions on signing
the Contributor License Agreement (CLA)
if you haven't done so.
Please follow the instructions to sign the CLA.
A `cla:yes` tag is then added to the pull request.

### Step 5. Automated tests
A set of automated tests will also start to run after creating the pull request.
If the tests fail, you may look into the error messages and try to fix it. 

### Step 6. Code review
A reviewer will review the pull request and provide comments.
There may be several rounds of comments and code changes
before the pull request gets approved by the reviewer.

### Step 7. Merge
Once the pull request is approved,
a `ready to pull` tag will be added to the pull request.
A team member will take care of the merging.

See the following images as an example for a PR and its related tests.

![PR and tests](pr_test.png)


## Setup environment

To setup the development environment,
We provide two options.
One is to use our Dockerfile, which builds into a container the required dev tools.
Another one is to setup a local environment by install the dev tools needed.

### Option 1: Use a Docker container

We provide a 
[Dockerfile](
https://github.com/keras-team/keras/blob/master/.devcontainer/Dockerfile)
to build the dev environment.
You can build the Dockerfile into a Docker image named `keras-dev`
with the following command at the root directory of your cloned repo.

```shell
docker build -t keras-dev .devcontainer
```

You can launch a Docker container from the image with the following command.
The `-it` option gives you an interactive shell of the container.
The `-v path/to/repo/:/home/keras/` mounts your cloned repo to the container.
Replace `path/to/repo` with the path to your cloned repo directory.

```shell
docker run -it -v path/to/repo/:/home/keras/ keras-dev
```

In the container shell, you need to install the latest dependencies
with the following command.

```shell
pip install -r /home/keras/requirements.txt
```

Now, the environment setup is complete. You are ready to run the tests.

You may modify the Dockerfile to your specific needs,
like installing your own dev tools.
You may also mount more volumes with the `-v` option, like your SSH credentials.
Besides the editors running in the shell,
many popular IDEs today also support developing in a container.
You may use these IDEs with the Dockerfile as well.

### Option 2: Setup a local environment

To setup your local dev environment, you will need the following tools.

1.  [Bazel](https://bazel.build/) is the tool to build and test Keras. See the
    [installation guide](https://docs.bazel.build/versions/4.0.0/install.html)
    for how to install and config bazel for your local environment.
2.  [git](https://github.com/) for code repository management.
3.  [python](https://www.python.org/) to build and code in Keras.

Using Apple Mac as an example (and linux will be very similar), the following
commands set up and check the configuration of a local workspace.

```shell
scottzhu-macbookpro2:~ scottzhu$ which bazel
/Users/scottzhu/bin/bazel

scottzhu-macbookpro2:~ scottzhu$ which git
/usr/local/git/current/bin/git

scottzhu-macbookpro2:~ scottzhu$ which python
/usr/bin/python

# Keras requires at least python 3.7
scottzhu-macbookpro2:~ scottzhu$ python --version
Python 3.9.6
```

A [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) is a
powerful tool to create a self-contained environment that isolates any change
from the system level config. It is highly recommended to avoid any unexpected
dependency or version issue.

```shell
scottzhu-macbookpro2:workspace scottzhu$ git clone https://github.com/keras-team/keras.git
Cloning into 'keras'...
remote: Enumerating objects: 492, done.
remote: Counting objects: 100% (492/492), done.
remote: Compressing objects: 100% (126/126), done.
remote: Total 35951 (delta 381), reused 443 (delta 366), pack-reused 35459
Receiving objects: 100% (35951/35951), 15.70 MiB | 16.09 MiB/s, done.
Resolving deltas: 100% (26243/26243), done.

scottzhu-macbookpro2:workspace scottzhu$ mkdir venv_dir
scottzhu-macbookpro2:workspace scottzhu$ python3 -m venv venv_dir
scottzhu-macbookpro2:workspace scottzhu$ source venv_dir/bin/activate
(venv_dir) scottzhu-macbookpro2:workspace scottzhu$ ls
keras       venv_dir

(venv_dir) scottzhu-macbookpro2:workspace scottzhu$ cd keras

(venv_dir) scottzhu-macbookpro2:workspace scottzhu$ pip install -r requirements.txt
Collecting pandas
  Using cached pandas-1.2.3-cp38-cp38-manylinux1_x86_64.whl (9.7 MB)
Collecting pydot
...
...
...

# Since tf-nightly uses keras-nightly as a dependency, we need to uninstall
# keras-nightly so that tests will run against keras code in local workspace.
(venv_dir) scottzhu-macbookpro2:workspace scottzhu$ pip uninstall keras-nightly
Found existing installation: keras-nightly 2.5.0.dev2021032500
Uninstalling keras-nightly-2.5.0.dev2021032500:
  Successfully uninstalled keras-nightly-2.5.0.dev2021032500
```

## Run tests

```shell
(venv_dir) scottzhu-macbookpro2:keras scottzhu$ bazel test -c opt keras:backend_test
WARNING: The following configs were expanded more than once: [v2]. For repeatable flags, repeats are counted twice and may lead to unexpected behavior.
INFO: Options provided by the client:
  Inherited 'common' options: --isatty=1 --terminal_columns=147
INFO: Reading rc options for 'test' from /Users/scottzhu/workspace/keras/.bazelrc:
  Inherited 'build' options: --apple_platform_type=macos --define open_source_build=true --define=use_fast_cpp_protos=false --define=tensorflow_enable_mlir_generated_gpu_kernels=0 --define=allow_oversize_protos=true --spawn_strategy=standalone -c opt --announce_rc --define=grpc_no_ares=true --config=short_logs --config=v2
INFO: Reading rc options for 'test' from /Users/scottzhu/workspace/keras/.bazelrc:
  'test' options: --define open_source_build=true --define=use_fast_cpp_protos=false --config=v2
INFO: Found applicable config definition build:short_logs in file /Users/scottzhu/workspace/keras/.bazelrc: --output_filter=DONT_MATCH_ANYTHING
INFO: Found applicable config definition build:v2 in file /Users/scottzhu/workspace/keras/.bazelrc: --define=tf_api_version=2 --action_env=TF2_BEHAVIOR=1
INFO: Found applicable config definition build:v2 in file /Users/scottzhu/workspace/keras/.bazelrc: --define=tf_api_version=2 --action_env=TF2_BEHAVIOR=1
INFO: Analyzed target //keras:backend_test (0 packages loaded, 0 targets configured).
INFO: Found 1 test target...
Target //keras:backend_test up-to-date:
  bazel-bin/keras/backend_test
INFO: Elapsed time: 45.535s, Critical Path: 45.26s
INFO: 19 processes: 19 local.
INFO: Build completed successfully, 20 total actions
//keras:backend_test                                                     PASSED in 45.2s
  Stats over 4 runs: max = 45.2s, min = 40.0s, avg = 41.5s, dev = 2.1s

INFO: Build completed successfully, 20 total actions
```
