#!/usr/bin/env bash

# Install System dependencies.
sudo apt-get -y update && apt-get -y install \
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
