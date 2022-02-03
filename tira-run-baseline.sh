#!/usr/bin/env bash

./tira-run.sh "$1" "$2" "grimjack-baseline" \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --stance-tagger sentiment \
  --stance-threshold 0.5
