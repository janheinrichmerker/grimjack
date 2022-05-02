#!/usr/bin/env bash

./tira-run.sh "$1" "$2" "grimjack-baseline" \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --quality-tagger debater \
  --stance-tagger debater-sentiment \
  --stance-threshold 0.125 \
  --num-hits 100
