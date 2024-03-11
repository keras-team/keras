Keras 3 is a high-velocity open-source project. We welcome contributions!

Contributions can be made in a variety of ways, including coding, enriching documentation, refining docstrings, and providing code examples.


## Current items open for contributions
At [this link](https://github.com/keras-team/keras/issues/18442), you'll find a list of items where you help is needed!


## How to contribute code

Follow these steps to submit your code contribution.

### Step 1. Open an issue

Before making any changes, we recommend opening an issue (if one doesn't already
exist) and discussing your proposed changes. This way, we can give you feedback
and validate the proposed changes.

If the changes are minor (simple bug fix or documentation fix), then feel free
to open a Pull Request (PR) without discussion.

### Step 2. Make code changes

To make code changes, you need to fork the repository. You will need to setup a
development environment and run the unit tests. This is covered in the section
"Setup environment".

### Step 3. Create a pull request

Once the change is ready, open a pull request from your branch in your fork to
the master branch in [keras-team/keras](https://github.com/keras-team/keras).

### Step 4. Sign the Contributor License Agreement

After creating the pull request, the `cla/google` check will be performed and,
if you haven't signed the Contributor License Agreement (CLA), it will fail with
instructions on how to do so. Please follow the instructions to sign the CLA and
the check will pass.

![CLA signed](https://github.com/keras-team/keras/assets/1091026/71c26353-e3b5-4135-8bae-64693c717775)


### Step 5. Code review

If the tests fail, look into the error messages and try to fix them.

![CI tests](https://github.com/keras-team/keras/assets/1091026/6f6c17ef-6bd7-4e95-9fbc-1906cde37380)

A reviewer will review the pull request and provide comments. There may be
several rounds of comments and code changes before the pull request gets
approved by the reviewer.

![Approval from reviewer](https://github.com/keras-team/keras/assets/1091026/8d28f74c-21e9-4146-b0ff-62d649a552a8)

### Step 6. Merging

Once the pull request is approved, a `ready to pull` tag will be added to the
pull request. A team member will take care of the merging.

![Ready to pull and merged](https://github.com/keras-team/keras/assets/1091026/c3908345-d7ae-44ee-a428-01f3b448b46b)

Here is an [example pull request](https://github.com/keras-team/keras/pull/18848)
for your reference.

## Setup environment

We provide two ways of setting up a development environment. One is to use a
dev container, and the other one is to set up a local environment by installing
the dev tools needed.

### Option 1: GitHub Codespace or dev container

We support GitHub Codespaces, Visual Studio Code dev containers and JetBrain dev
containers. Please see the
[Dev container documentation](https://github.com/keras-team/keras/tree/master/.devcontainer).

### Option 2: Set up a local environment

To set up your local dev environment, you will need the following tools.

1.  [git](https://github.com/) for code repository management.
2.  [python](https://www.python.org/) to build and code in Keras.

The following commands check the tools above are successfully installed. Note
that Keras requires at least Python 3.9 to run.

```shell
git --version
python --version
```

Clone your forked repo to your local machine. Go to the cloned directory to
install the dependencies.

```shell
git clone https://github.com/YOUR_GITHUB_USERNAME/keras.git
cd keras
pip install -r requirements.txt
```

You then need to configure the backend to use, see the
[Configuring your backend](https://github.com/keras-team/keras/blob/master/README.md#configuring-your-backend)
section of the README.

You can also add GPU support to your environment, see the
[Adding GPU support](https://github.com/keras-team/keras/blob/master/README.md#adding-gpu-support)
section of the README.

## Code style

Keras uses [Black](https://black.readthedocs.io/en/stable/) and
[isort](https://pycqa.github.io/isort/) to format the code. Please refer to
[requirements-common.txt](https://github.com/keras-team/keras/blob/master/requirements-common.txt)
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
[(link)](https://github.com/keras-team/keras/blob/v3.0.0/keras/layers/attention/multi_head_attention.py#L20).

A **function docstring** may contain the following items:

* One-line description of the function.
* Paragraph(s) of more detailed information.
* Optional `Examples` section.
* `Args` section for the function arguments.
* `Returns` section for the return values.
* Optional `Raises` section for possible errors.

You can check out `text_dataset_from_directory` as an example
[(link)](https://github.com/keras-team/keras/blob/v3.0.0/keras/utils/text_dataset_utils.py#L27).

## Run tests

We use [pytest](https://pytest.org/) to run the tests.

### Run a test file

To run the tests in `keras/losses/losses_test.py`, use the following command
at the root directory of the repo.

```shell
pytest keras/losses/losses_test.py
```

### Run a single test case

You can specify a single test class to run within a file.

```shell
pytest keras/losses/losses_test.py::MeanSquaredErrorTest
```

You can also specify a single test method to run within a class.

```shell
pytest keras/losses/losses_test.py::MeanSquaredErrorTest::test_sample_weighted
```

### Run all tests

You can run all the tests locally by running the following command in the repo
root directory.

```shell
pytest keras
```

Note that you can skip the Keras applications tests using the
`SKIP_APPLICATIONS_TESTS` environment variable. This will cut down the testing
time significantly.

```shell
SKIP_APPLICATIONS_TESTS=True pytest keras
```

To run all tests using a different backend, you can simply specify it on the
command line.

```shell
KERAS_BACKEND=jax SKIP_APPLICATIONS_TESTS=True pytest keras
```
