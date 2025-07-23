#! /bin/bash

pytest --collect-only -q | grep "<Module " | awk '{print $2}' | sed 's/>//' | sort -u | xargs -I {} find . -name "{}" -print
