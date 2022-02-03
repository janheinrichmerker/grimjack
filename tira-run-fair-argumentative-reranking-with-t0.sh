#!/usr/bin/env bash

./tira-run.sh "$1" "$2" "grimjack-fair-argumentative-reranking-with-t0" \
  --query-expander t0pp-description-narrative \
  --query-expander t0pp-synonyms \
  --query-expander fast-text-wiki-news-synonyms \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --quality-tagger debater \
  --stance-tagger debater-sentiment \
  --stance-threshold 0.125 \
  --num-hits 20 \
  --rerank-hits 10 \
  --reranker axiomatic \
  --argumentative-axioms \
  --reranker subjective-first \
  --reranker alternating-stance
