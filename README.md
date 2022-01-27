[![CI](https://img.shields.io/github/workflow/status/heinrichreimer/grimjack/CI?style=flat-square)](https://github.com/heinrichreimer/grimjack/actions?query=workflow%3A"CI")
[![Code coverage](https://img.shields.io/codecov/c/github/heinrichreimer/grimjack?style=flat-square)](https://codecov.io/github/heinrichreimer/grimjack/)
[![Issues](https://img.shields.io/github/issues/heinrichreimer/grimjack?style=flat-square)](https://github.com/heinrichreimer/grimjack/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/heinrichreimer/grimjack?style=flat-square)](https://github.com/heinrichreimer/grimjack/commits)
[![License](https://img.shields.io/github/license/heinrichreimer/grimjack?style=flat-square)](LICENSE)

# 🤺 grimjack

Argumentative passage search engine, named after the fencer [Grimjack](https://en.wikipedia.org/wiki/Grimjack).

Participation in the [Touché 2022](https://webis.de/events/touche-22/) shared task 2, as part of the
[Advanced Information Retrieval](https://gitlab.informatik.uni-halle.de/aqvbw/Information-Retrieval/) lecture
at [Martin Luther University Halle-Wittenberg](https://uni-halle.de).

## Usage

The following sections describe how to use the Grimjack search engine using Pipenv.
See the [Docker section](#docker) for instructions on how to use Grimjack inside a Docker container.

### Installation

First, install [Python 3](https://python.org/downloads/),
[pipx](https://pipxproject.github.io/pipx/installation/#install-pipx), and
[Pipenv](https://pipenv.pypa.io/en/latest/install/#isolated-installation-of-pipenv-with-pipx).
Then install dependencies (this may take a while):

```shell script
pipenv install
```

### Run the search pipeline

To test the search pipeline, run the `grimjack` CLI like this:

```shell script
pipenv run python -m grimjack search "Which is better, a laptop or a desktop?"
```

### Generate a run file for all topics

To search all topics and generate a run file (top-5 per query), run the `grimjack` CLI like this:

```shell script
pipenv run python -m grimjack --num-hits 5 run data/run.txt
```

This will save a run file at `data/run.txt` with the format described in
the [shared task description](https://webis.de/events/touche-22/shared-task-2.html#submission).

### Evaluate all topics

To evaluate the search pipeline for all topics, run the `grimjack` CLI like this:

```shell script
pipenv run python -m grimjack evaluate
```

This will print the evaluation metric (default: nDCG@10) to the console.

### Options

The search pipeline can be configured with the options listed in the help command. The help command also lists all
subcommands.

```shell script
pipenv run python -m grimjack --help
```

Each subcommand's extra options can be listed, e.g.:

```shell script
pipenv run python -m grimjack search --help
```

### Examples / Touché 2022 Runs

The following examples correspond to the runs we submit to the
[Touché 2022 shared task](https://webis.de/events/touche-22/).

#### 1. Baseline

```shell
pipenv run python -m grimjack \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --stance-tagger sentiment \
  --stance-threshold 0.5 \
  --num-hits 100 \
  run \
  data/runs/grimjack-baseline.txt
```

#### 2. All you need is T0

TODO

```shell
pipenv run python -m grimjack \
  --retrieval-model query-likelihood-dirichlet \
  --query-expander t0pp-synonyms \
  --query-expander t0pp-description-narrative \
  --targer-model tag-ibm-fasttext \
  --quality-tagger t0pp \
  --stance-tagger t0pp \
  --stance-threshold 0.5 \
  --num-hits 100 \
  run \
  data/runs/grimjack-all-you-need-is-t0.txt
```

#### 3. Argumentative Axioms

TODO

```shell
pipenv run python -m grimjack \
  --retrieval-model query-likelihood-dirichlet \
  --reranker axiomatic \
  --rerank-hits 20 \
  --argumentative-axioms \
  --targer-model tag-ibm-fasttext \
  --stance-tagger sentiment \
  --stance-threshold 0.5 \
  --num-hits 100 \
  run \
  data/runs/grimjack-argumentative-axioms.txt
```

#### 4. Fair Reranking

TODO

```shell
pipenv run python -m grimjack \
  --retrieval-model query-likelihood-dirichlet \
  --reranker balanced-top-5-stance \
  --targer-model tag-ibm-fasttext \
  --stance-tagger sentiment \
  --stance-threshold 0.5 \
  --num-hits 100 \
  run \
  data/runs/grimjack-fair-reranking.txt
```

#### 5. Argumentative Fair Reranking with T0

TODO

```shell
pipenv run python -m grimjack \
  --retrieval-model query-likelihood-dirichlet \
  --query-expander fast-text-wiki-news-synonyms \
  --query-expander t0pp-synonyms \
  --query-expander t0pp-description-narrative \
  --reranker axiomatic \
  --reranker balanced-top-5-stance \
  --rerank-hits 20 \
  --targer-model tag-ibm-fasttext \
  --stance-tagger sentiment \
  --stance-threshold 0.5 \
  --num-hits 100 \
  run \
  data/runs/grimjack-argumentative-fair-reranking-with-T0.txt
```

### 6. Fine tune based on Touché 2020/2021 qrels

## Testing

After [installing](#installation) all dependencies, you can run all unit tests:

```shell script
pipenv run pytest
```

## Docker

Grimjack can also be used as a Docker container:

```shell
docker image build . -t grimjack
docker container run grimjack --help
```

We recommend to bind mount the data directory to the container, for example:

```shell
docker container run -v "$(pwd)"/data:/workspace/data grimjack run data/run.txt
```

## License

This repository is licensed under the [MIT License](LICENSE).
