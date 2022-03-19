#!/usr/bin/env bash

if [[ $UID != 0 ]]; then
  echo "Please run this script with sudo:"
  echo "sudo $0 $*"
  exit 1
fi

# Install System dependencies.
apt-get -y update && apt-get -y install \
  git \
  openjdk-11-jdk \
  python3.9 \
  python3.9-venv \
  python3-pip \
  python3.9-dev

# Install Python dependencies.
python3.9 -m pip install --upgrade pip
python3.9 -m pip install --user pipx
pipx install pipenv

# Install project dependencies.
pipenv install --deploy
