[![CI](https://img.shields.io/github/actions/workflow/status/janheinrichmerker/grimjack/ci.yml?branch=main&style=flat-square)](https://github.com/janheinrichmerker/grimjack/actions/workflows/ci.yml)
[![Code coverage](https://img.shields.io/codecov/c/github/janheinrichmerker/grimjack?style=flat-square)](https://codecov.io/github/janheinrichmerker/grimjack/)
[![Issues](https://img.shields.io/github/issues/janheinrichmerker/grimjack?style=flat-square)](https://github.com/janheinrichmerker/grimjack/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/janheinrichmerker/grimjack?style=flat-square)](https://github.com/janheinrichmerker/grimjack/commits)
[![License](https://img.shields.io/github/license/janheinrichmerker/grimjack?style=flat-square)](LICENSE)

# 🤺 grimjack

Argumentative passage search engine, named after the fencer [Grimjack](https://en.wikipedia.org/wiki/Grimjack).

Participation in the [Touché 2022](https://webis.de/events/touche-22/) shared task 2, as part of the
[Advanced Information Retrieval](https://gitlab.informatik.uni-halle.de/aqvbw/Information-Retrieval/) lecture
at [Martin Luther University Halle-Wittenberg](https://uni-halle.de).

## Usage

The following sections describe how to use the Grimjack search engine using Pipenv.
See the [Docker section](#docker) for instructions on how to use Grimjack inside a Docker container.

### Installation

First, install [Python 3.9](https://python.org/downloads/) or higher and then clone this repository.
From inside the repository directory, create a virtual environment and activate it:

```shell
python3.9 -m venv venv/
source venv/bin/activate
```

Then, install the test dependencies:

```shell
pip install -e .
```

### Run the search pipeline

To test the search pipeline, run the `grimjack` CLI like this:

```shell script
python -m grimjack search "Which is better, a laptop or a desktop?"
```

### Generate a run file for all topics

To search all topics and generate a run file (top-5 per query), run the `grimjack` CLI like this:

```shell script
python -m grimjack --num-hits 5 run data/run.txt
```

This will save a run file at `data/run.txt` with the format described in
the [shared task description](https://webis.de/events/touche-22/shared-task-2.html#submission).

### Evaluate all topics

To evaluate the search pipeline for all topics, run the `grimjack` CLI like this:

```shell script
python -m grimjack evaluate
```

This will print the evaluation metric (default: nDCG@10) to the console.

### Options

The search pipeline can be configured with the options listed in the help command. The help command also lists all
subcommands.

```shell script
python -m grimjack --help
```

Each subcommand's extra options can be listed, e.g.:

```shell script
python -m grimjack search --help
```

### Examples / Touché 2022 Runs

The following examples correspond to the runs we submit to the
[Touché 2022 shared task](https://webis.de/events/touche-22/).

#### 1. Baseline

Retrieve 20 documents by Dirichlet query likelihood for the unmodified query, 
tag arguments using the IBM fastText TARGER model,
tag argument quality using the IBM Debater API,
tag argument stance by comparing sentiments for each object using the IBM Debater API,
treating stance as neutral if under a threshold of 0.125.

```shell
python -m grimjack \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --quality-tagger debater \
  --stance-tagger debater-sentiment \
  --stance-threshold 0.125 \
  --num-hits 100 \
  run \
  data/runs/grimjack-baseline.txt
```

#### 2. Argumentative Axioms

Rerank top-10 documents from the baseline result 
based on preferences from argumentative axioms.

```shell
python -m grimjack \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --quality-tagger debater \
  --stance-tagger debater-sentiment \
  --stance-threshold 0.125 \
  --num-hits 100 \
  --rerank-hits 10 \
  --reranker axiomatic \
  --argumentative-axioms \
  run \
  data/runs/grimjack-argumentative-axioms.txt
```

#### 3. Fair Reranking after Argumentative Axioms

With the argumentative axiomatic ranking,
move subjective documents (non-neutral stance) to the top,
and then ensure that document's stances alternate.

```shell
python -m grimjack \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --quality-tagger debater \
  --stance-tagger debater-sentiment \
  --stance-threshold 0.125 \
  --num-hits 100 \
  --rerank-hits 10 \
  --reranker axiomatic \
  --argumentative-axioms \
  --reranker subjective-first \
  --reranker alternating-stance \
  run \
  data/runs/grimjack-fair-reranking-argumentative-axioms.txt
```

#### 4. All you need is T0

Expand the query by extracting queries from the description and narrative using T0, 
expand the query with T0 synonyms for each comparative object,
then retrieve like with the baseline,
tag argument quality using T0, tag argument stance using T0.

```shell
python -m grimjack \
  --query-expander t0pp-description-narrative \
  --query-expander t0pp-comparative-synonyms \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --quality-tagger t0pp \
  --stance-tagger t0pp \
  --stance-threshold 0.125 \
  --num-hits 100 \
  run \
  data/runs/grimjack-all-you-need-is-t0.txt
```

#### 5. Argumentative Fair Reranking with T0

Expand the query by extracting queries from the description and narrative using T0, 
expand the query with T0 and fastText synonyms for each comparative object,
then retrieve like with the baseline,
rerank top-10 documents from the baseline result 
based on preferences from argumentative axioms,
move subjective documents (non-neutral stance) to the top,
and then ensure that document's stances alternate.

```shell
python -m grimjack \
  --query-expander t0pp-description-narrative \
  --query-expander t0pp-synonyms \
  --query-expander fast-text-wiki-news-synonyms \
  --retrieval-model query-likelihood-dirichlet \
  --targer-model tag-ibm-fasttext \
  --quality-tagger debater \
  --stance-tagger debater-sentiment \
  --stance-threshold 0.125 \
  --num-hits 100 \
  --rerank-hits 10 \
  --reranker axiomatic \
  --argumentative-axioms \
  --reranker subjective-first \
  --reranker alternating-stance \
  run \
  data/runs/grimjack-fair-argumentative-reranking-with-t0.txt
```

## Testing

After [installing](#installation) all dependencies, you can run all unit tests:

```shell script
flake8 grimjack
pylint -E grimjack
pytest grimjack
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

## Ideas

- Fairness
- Submit to ArgMining workshop
