#!/bin/bash
sudo pip install -r requirements.txt
sudo pip uninstall keras-nightly -y

wget https://github.com/cli/cli/releases/download/v2.17.0/gh_2.17.0_linux_amd64.deb -P /tmp
sudo apt install /tmp/gh_2.17.0_linux_amd64.deb -y