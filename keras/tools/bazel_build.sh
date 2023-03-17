#!/bin/bash
BAZEL_VERSION=5.4.0
rm -rf ~/bazel
mkdir ~/bazel

pushd ~/bazel
wget https://github.com/bazelbuild/bazel/releases/download/"${BAZEL_VERSION}"/bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh
chmod +x bazel-*.sh
./bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh --user
rm bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh
popd

PATH="/home/kbuilder/bin:$PATH"
which bazel
bazel version

TAG_FILTERS="-no_oss,-oss_excluded,-oss_serial,-gpu,-benchmark-test,-no_oss_py3,-no_pip,-nopip"
bazel build \
    --define=use_fast_cpp_protos=false \
    --build_tag_filters="${TAG_FILTERS}" \
    -- //keras/...
