#!/bin/bash
if [ -z "$1" ]; then
    echo "Please provide the path to the directory containing the 'requirements.in' file"
    exit 1
fi

pip-compile "$1/requirements.in" -q
