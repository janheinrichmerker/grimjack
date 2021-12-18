#!/usr/bin/env bash

# Parse options.
input_dir=$1
output_dir=$2

# Install System dependencies.
apt-get -y update && apt-get -y install git openjdk-11-jdk
pip install --upgrade pip && pip install pipenv

# Ensure correct Git revision.
git pull
git checkout 4c81795877ed1ac9c1fc6895896a2ae333ff35f8

# Install project dependencies.
pipenv install

# Copy topic file.
topic_file_name="topics.xml"
topic_url="ALREADY_DOWNLOADED"
topic_url_hash=$(echo -n $topic_url | md5sum | awk '{print $1}')
topic_dir="./data/topics/$topic_url_hash"
mkdir -p "$topic_dir"
cp "$input_dir/$topic_file_name" "$topic_dir/$topic_file_name"

# Generate run file.
run_name="grimjack-baseline"
run_file="data/runs/$run_name.txt"
pipenv run python -m grimjack \
  --topics-url "$topic_url" \
  --topics-path "$topic_file_name" \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --stance-tagger sentiment \
  --stance-threshold 0.5 \
  --num-hits 100 \
  run \
  "$run_file"

# Copy run file.
cp "$run_file" "$output_dir/run.txt"