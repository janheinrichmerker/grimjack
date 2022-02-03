#!/usr/bin/env bash

./tira-run.sh "$1" "$2" "grimjack-all-you-need-is-t0" \
  --query-expander t0pp-description-narrative \
  --query-expander t0pp-comparative-synonyms \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --quality-tagger t0pp \
  --stance-tagger t0pp \
  --stance-threshold 0.125 \
  --num-hits 20
