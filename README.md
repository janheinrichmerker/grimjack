[![CI](https://img.shields.io/github/workflow/status/heinrichreimer/grimjack/CI?style=flat-square)](https://github.com/heinrichreimer/grimjack/actions?query=workflow%3A"CI")
[![Code coverage](https://img.shields.io/codecov/c/github/heinrichreimer/grimjack?style=flat-square)](https://codecov.io/github/heinrichreimer/grimjack/)
[![Issues](https://img.shields.io/github/issues/heinrichreimer/grimjack?style=flat-square)](https://github.com/heinrichreimer/grimjack/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/heinrichreimer/grimjack?style=flat-square)](https://github.com/heinrichreimer/grimjack/commits)
[![License](https://img.shields.io/github/license/heinrichreimer/grimjack?style=flat-square)](LICENSE)

# 🤺 grimjack

Named after the fencer [Grimjack](https://en.wikipedia.org/wiki/Grimjack).

Participation in the [Touché 2022](https://webis.de/events/touche-22/) shared task 2, as part of the 
[Advanced Information Retrieval](https://gitlab.informatik.uni-halle.de/aqvbw/Information-Retrieval/) lecture.

## Usage

### Installation

First, install [Python 3](https://python.org/downloads/), 
[pipx](https://pipxproject.github.io/pipx/installation/#install-pipx), and 
[Pipenv](https://pipenv.pypa.io/en/latest/install/#isolated-installation-of-pipenv-with-pipx).
Then install dependencies (may take a while):

```shell script
pipenv install
```

### Run the search pipeline

To test the search pipeline, run the `grimjack` CLI like this:

```shell script
pipenv run python -m grimjack search "Which is better, a laptop or a desktop?"
```

### Generate a run file for all topics

To search all topics and generate a run file, run the `grimjack` CLI like this:

```shell script
pipenv run python -m grimjack run-all data/run.txt
```

This will save a run file at `data/run.txt` with the format described in
the [shared task description](https://webis.de/events/touche-22/shared-task-2.html#submission).

#### Options

The search pipeline can be configured with the options listed in the help command.
The help command also lists all subcommands.

```shell script
pipenv run python -m grimjack --help
```

Each subcommand's extra options can be listed like this:

```shell script
pipenv run python -m grimjack search --help
```

#### Examples

Search for the top 3 results with query expansion using synonyms from the T0pp language model 
and no stemming for the index.

```shell script
pipenv run python -m grimjack --no-stemmer --query-expansion t0pp search -k 3 "Which is better, a laptop or a desktop?"
```

### Testing

Run all unit tests:

```shell script
pipenv run pytest
```

### Docker

Grimjack can also be used as a Docker container:

```shell
docker image build . -t grimjack
docker container run grimjack --help
```

## License

This repository is licensed under the [MIT License](LICENSE).
