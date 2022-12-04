FROM mcr.microsoft.com/vscode/devcontainers/python:3.8
COPY setup.sh /setup.sh

# Install Bazel
RUN sudo apt install wget -y
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-amd64
RUN chmod a+x bazelisk-linux-amd64
RUN mv bazelisk-linux-amd64 /usr/bin/bazel